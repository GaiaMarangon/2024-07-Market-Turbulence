clc
clear
close all

%--- LOAD DATA -----------------------------------------------------------

% Load data, dates and returns
 [dates,data,returns,dataMarket,returnsMarket] = loadData();

%--- SET PARAMETERS ------------------------------------------------------

% Dimensions
numAssets   = size(data, 2);
numWeeks    = 52;
windowSize  = 78;
scaleFactor = 52;   % nr of single periods (weeks) in a year, for annualized quantities

% Probability level for CK inliers, outliers
alpha     = 0.05;       

% Select the method for HMM:
% hmmMethod = 1;  % Discretization of observations
hmmMethod = 2;  % Normal distribution for the observations


% HMM Initial guesses (for first loop) (ONLY FOR DISCRETIZED OBSERVATIONS VERSION)
% - Prior 
guessPrior = [.8,.2];
% - Transition probability matrix (larger probability to stay than to switch)
guessTr = [.8, .2; .2, .8];
% - Observations probability matrix (different expected means for stress and calm)
guessEm = [.05,.5,.2,.2,.05; .05,.2,.2,.5,.05];


% Optimization Problem Parameters
% - Weights bounds
MinWeight = 0;     % Minimum weight for assets
MaxWeight = 1;     % Maximum weight for assets
% - Tracking error constraint
trackingErrorLimit = 0.025; % tested alternative: 0.015
% - Initial guess for portfolio weights (first loop)
w0 = ones(numAssets,1)/numAssets;                    


% Indicators Parameters
% - Annual risk free rate (based on data before starting the analysis)
delta1 = windowSize;                % on restricted window
annualRiskFreeRate1 =  prod(returns(end-numWeeks-delta1+1:end-numWeeks,2)+1)^(scaleFactor/delta1) -1;

% % (OPTIONAL) Compare with different choice
% delta2 = length(returns)-numWeeks;  % on all available historical data 
% annualRiskFreeRate2 =  prod(returns(end-numWeeks-delta2+1:end-numWeeks,2)+1)^(scaleFactor/delta2) -1;
% fprintf("Compare Risk Free Rate: on resticted window = %.4f on all data = %.4f\n",annualRiskFreeRate1,annualRiskFreeRate2)
annualRiskFreeRate  = annualRiskFreeRate1;

% - Weekly risk free rate
weeklyRiskFree = (1+annualRiskFreeRate)^(1/numWeeks) -1;

% - Minimium Acceptable Return (for Downside Std Dev)
MAR = weeklyRiskFree;    

% - Betas (of each asset, based on data before starting the analysis)
betas = zeros(numAssets,1);
for i=1:numAssets
    betas(i) = returnsMarket(end-numWeeks-delta1+1:end-numWeeks) \ returns(end-numWeeks-delta1+1:end-numWeeks,i);
end

% - Market quotes 
quotesMarket = cumprod(1+returnsMarket(end-numWeeks+1:end))*100; 

% Benchmark weights
wBench = [
    0.16; % Euro Area Government Bond
    0.09; % United States Government Bond
    0.06; % Japan Government Bond
    0.03; % United Kingdom Government Bond
    0.02; % Global Emerging Markets Bonds
    0.05; % Euro Area Corporate Bonds
    0.03; % United States Corporate Bonds
    0.01; % Japan Corporate Bonds
    0.01; % United Kingdom Corporate Bonds
    0.01; % Global High Yield Bonds
    0.16; % Euro Area Equity
    0.12; % Europe Ex Euro Area Equity
    0.08; % Asia/Pacific Equity
    0.12; % North America Equity
    0.05  % Emerging Markets Equity
];

%--- COMPUTE HISTORICAL INFORMATION --------------------------------------

% Define time window which will be used during the analysis (analyzed weeks and rolling windows)
refReturns = returns(end-numWeeks-windowSize:end,:);

% Plot Historical Returns
figure()
newcolors = parula(numAssets);
colororder(newcolors)
hold on
for i=1:numAssets
    plot(refReturns(:,i),"DisplayName","asset "+num2str(i))
end
xlabel("Time (w)")
ylabel("Asset Returns")
legend("Location","eastoutside")

% Compute Market turbulence measure
[~, ~, distVec] = CK_inoutliers(refReturns, alpha);
% Plot Market turbulence measure
figure()
plot(distVec)
yline(chi2inv(1-alpha,size(returns,2)),color="red")
xline(windowSize,color="black")
xlabel("Time (w)")
ylabel("d^2")
legend("Observed squared distance","Threshold", "Location","best")
title("Market turbulence measure")

%--- PREALLOCATE RESULTS -------------------------------------------------

% Estimated sequence of states (calm or stress)
estStatesVec = zeros(numWeeks,1);

% Weight matrix
wMatrix = zeros(numWeeks, numAssets);

% Performance (time series, weekly returns)
returnsBench  = zeros(numWeeks,1); 
returnsPort   = zeros(numWeeks,1); 
activeReturns = zeros(numWeeks,1);

% Quotes (time series, weekly quotes)
quotesBench = zeros(numWeeks,1);
quotesPort  = zeros(numWeeks,1);

% Total return (time series, not annualized)
totRetBench  = zeros(numWeeks,1);
totRetPort   = zeros(numWeeks,1);
totRetActive = zeros(numWeeks,1);

% Average return (single period, computed cumulatively on data)
avgRetBench  = zeros(numWeeks,1);
avgRetPort   = zeros(numWeeks,1);
avgActiveRet = zeros(numWeeks,1);
avgRiskFreeRet = zeros(numWeeks,1);

% Volatility (annualized, computed cumulatively on data)
volatBench     = zeros(numWeeks,1);
volatPort      = zeros(numWeeks,1);
volatActive    = zeros(numWeeks,1);
downsideStdDev = zeros(numWeeks,1);
% Betas
betaPort       = zeros(numWeeks,1);

% Indicators (annualized, computed cumulatively on data)
infoRatio    = zeros(numWeeks,1);
sharpeRatio  = zeros(numWeeks,1);
sortinoRatio = zeros(numWeeks,1);
treynorRatio = zeros(numWeeks,1);

%--- LOOP OVER WEEKS -----------------------------------------------------
for t = 1:numWeeks

    %--- 1. DEFINITION OF REFERENCE DATATBASE ----------------------------

    % Define historical window 
    endIdx = length(returns) - numWeeks + t -1; 
    startIdx = endIdx - windowSize + 1      ;    
                                            
    % Define reference database
    refReturns = returns(startIdx:endIdx, :);

    %--- 2. DEFINITION OF INLIERS AND OUTLIERS -------------------------
    
    % Identify inliers and outliers using Chow-Kritzman theory
    [inliers, outliers, distVec] = CK_inoutliers(refReturns, alpha);

    % Compute mean and covariance for inliers subsample
    meanInliers = mean(inliers);
    covInliers  = cov(inliers);
    % Compute mean and covariance for outliers subsample
    meanOutliers = mean(outliers);
    covOutliers  = cov(outliers);

    %--- 3. HIDDEN MARKOV FOR DISTRESS PROBABILITY -----------------------

    if hmmMethod==1
        %--- VERSION 1 (DISCRETIZED OBSERVATIONS)

        % Estimate distress probability, according to HMM 
        [estTr,estEm,estPrior,estStates,p] = distressHMM(distVec,guessTr,guessEm,guessPrior);
        % Update initial guess to current estimates
        guessTr = estTr;
        guessEm = estEm;
        guessPrior = estPrior;
        % Save estimated current state 
        estStatesVec(t) = estStates(end);

        % % (OPTIONAL) For last loop, print estimated matrices and plot estimated states
        % if t==numWeeks
        %     fprintf("\nEstimated Transition Matrix:\n")
        %     fprintf([repmat(' %.4f',1,2),'\n'],estTr')
        %     fprintf("\nEstimated Emission Matrix:\n")
        %     fprintf([repmat(' %.4f',1,5),'\n'],estEm')
        %     fprintf("\n")
        % 
        %     figure()
        %     plot(estStates,'LineStyle','-','Marker','.')
        %     xlabel("Time (w)")
        %     ylabel("State")
        %     ylim([0,3])
        %     title("Estimated States")
        % end

        %-------------------------
    else
        %--- VERSION 2 (NORMAL OBSERVATIONS) 

        % Estimate distress probability, according to HMM (version 2)
        [TransProb, mu, sigma, pCurr, smoothed] = fithmm(distVec);
        if pCurr(2) == 1.00
            p = TransProb(2, 1);
        else 
            p = TransProb(1, 1);
        end

        % Save estimated current state
        [~,sCurr] = max(pCurr);
        estStatesVec(t) = sCurr;

        % % (OPTIONAL) Check that probability of negative values is negligible
        % fprintf("Prob (percent) of neg values for state 1: %.4f, for state 2: %.4f\n",normcdf(0,mu(1),sigma(1))*100,normcdf(0,mu(1),sigma(1))*100)
        
        % % (OPTIONAL) For last loop, print estimated matrices and plot estimated states
        % if t==numWeeks
        %     [~,estStates] = max(smoothed,[],2);
        %     fprintf("\nEstimated Transition Matrix:\n")
        %     fprintf([repmat(' %.4f',1,2),'\n'],TransProb')
        %     fprintf("\nEstimated Means:\n")
        %     fprintf([repmat(' %.4f',1,2),'\n'],mu)
        %     fprintf("\nEstimated Variances:\n")
        %     fprintf([repmat(' %.4f',1,2),'\n'],sigma)
        %     fprintf("\n")
        % 
        %     figure()
        %     plot(estStates,'LineStyle','-','Marker','.')
        %     xlabel("Time (w)")
        %     ylabel("State")
        %     ylim([0,3])
        %     title("Estimated States")
        % end
        %------------------------------
    end

    %--- 4. ESTIMATION OF MIXED OPTIMIZATION PARAMETERS ------------------

    % Define Chow-Kritzman mean and covariance
    meanCK = (1-p)*meanInliers + p*meanOutliers;
    covCK  = (1-p)*covInliers  + p*covOutliers;

    %--- OPTIMIZATION WITH CONSTRAINTS -----------------------------------

    % Objective function: negative Information Ratio
    f = @(w) -( meanCK*(w-wBench) / sqrt( (w-wBench)' * covCK * (w-wBench) ) ); 

    % Initial guess: uniform for first loop, or equal to previous optimal weights for subsequent loops
    
    % Constraints
    % - full-investment (sum(w)=1)
    Aeq = ones(1, numAssets);
    beq = 1;
    % - non negativity (0 <= w <= 1)
    lb = ones(numAssets, 1)*MinWeight;
    ub = ones(numAssets, 1)*MaxWeight;
    % - non-linear tracking error constraint ( (w-wB)'Sigma(w-wB)<= trErrLim^2 )
    nonlincon = @(w) deal([], scaleFactor * (w-wBench)' * covCK * (w-wBench) - trackingErrorLimit^2 );

    % Options
    options = optimset('Display', 'off');
    
    % Optimize portfolio
    wOpt = fmincon(f, w0, [], [], Aeq, beq, lb, ub, nonlincon, options);

    %--- 6. COLLECTION OF OPTIMAL ALLOCATIONS ----------------------------

    % Optimal weights
    wMatrix(t, :) = wOpt';
    % Update initial guess for weights
    w0 = wOpt;

    %--- 7. COMPUTE PERFORMACES ------------------------------------------

    % Here we evaluate the performace of the optimal weights computed up to time t (previous loops and up to current one)
    % Each optimal weight, computed at t, acts on returns at t+1 (if loop has gone through 1,...,t test performance in times 2,...,t+1)

    % Define windows of returns (up to time t)
    refReturnsCum=returns(length(returns)-numWeeks+1:endIdx+1,:);

    %--- Compute Benchmark performance

    % Performance (returns time series)
    returnsBench(t) = refReturnsCum(t,:)*wBench;   
    returnsBenchCum = returnsBench(1:t);
    % Quotes (percent time series)
    quotesBench(t) = prod(1+returnsBenchCum)*100; 
    
    % Total return (on numWeek)
    totRetBench(t) = quotesBench(t) /100 -1;    
    % Average return (single period)
    avgRetBench(t) = (1+totRetBench(t) )^(1/numWeeks) -1; 
    
    % Absolute risk (volatility, annualized)
    volatBench(t) = std(returnsBenchCum)  * sqrt(scaleFactor);
    
    %--- Collect indicators 

    % Performace (returns time series)
    returnsPort(t) = refReturnsCum(t,:)*wOpt;    
    returnsPortCum = returnsPort(1:t);
    % Active returns (time series)
    activeReturns(t) = returnsPort(t) - returnsBench(t);

    % Quotes (percent time series)
    quotesPort(t) = prod(1+returnsPortCum)*100; 

    % Total return (up to time t)
    totRetPort(t) = quotesPort(t) /100 -1;  
    % Total active return (up to time t)
    totRetActive(t) = totRetPort(t) - totRetBench(t);

    % Average return (single period)
    avgRetPort(t) = (1+totRetPort(t) )^(1/numWeeks) -1;         
    % Average active return (single period)
    avgActiveRet(t) = avgRetPort(t) - avgRetBench(t); 
    % Average return difference from risk free (single period)
    avgRiskFreeRet(t) = avgRetPort(t) - weeklyRiskFree; 

    % Absolute risk (volatility, annualized)
    volatPort(t) = std(returnsPortCum)  * sqrt(scaleFactor); 
    % Relative risk (Tracking error volatility)
    volatActive(t) = std(activeReturns(1:t)) * sqrt(scaleFactor); 
    % Downside Standard Deviation
    downsideStdDev(t) = std( returnsPortCum(returnsPortCum<MAR) -MAR );
    % Beta of Portfolio
    betaPort(t) = betas' * wOpt;

    % Information Ratio
    infoRatio(t)    = ( (avgActiveRet(t)  +1)^scaleFactor -1 ) / volatActive(t);
    % Sharpe Ratio
    sharpeRatio(t)  = ( (avgRiskFreeRet(t)+1)^scaleFactor -1 ) / volatPort(t);
    % Sortino Ratio
    sortinoRatio(t) = ( (avgRiskFreeRet(t)+1)^scaleFactor -1 ) / downsideStdDev(t);
    % Treynor Ratio
    treynorRatio(t) = ( (avgRiskFreeRet(t)+1)^scaleFactor -1 ) / betaPort(t);
      
end

%--- SAVE OR LOAD PERFORMANCES -------------------------------------------

% With Method 2 (Main Text), results are saved 
% With Method 1 (Supplementary), results of Method 2 are loaded for comparison 

if hmmMethod==2
    save("Indicators1.mat","returnsPort","quotesPort","infoRatio","sharpeRatio","sortinoRatio","treynorRatio")
    Temp = [];
else
    Temp = load("Indicators1.mat");
end

%--- REPORT PORTFOLIO PERFORMANCES (PLOTS) -------------------------------

% Portfolio composition
figure()
newcolors = parula(numAssets);
colororder(newcolors)
%
subplot(1,4,1:3)
area(wMatrix)
xlabel("Time (w)")
xlim([1,size(wMatrix,1)])
ylim([0,1])
title("Portfolio")
%
subplot(1,4,4)
area(repmat(wBench,1,2)')
title("Benchmark")

% Time series of Estimated states
figure()
hold on
plot(estStatesVec)
legend("Estimated states","Location","southeast")
xlabel("Time (w)")
ylabel("State")
ylim([0,3])
title("Time series of estimated states")

% Performance
if hmmMethod == 2
    % Performace
    figure()
    bar([returnsPort,returnsBench],'EdgeColor','none')
    yline(weeklyRiskFree)
    legend("Portfolio","Benchmark","Risk Free Rate")
    xlabel("Time (w)")
    ylabel("Performance")
    title("Performance")
else
% Performace comparison
    figure()
    bar([returnsPort,returnsBench,Temp.returnsPort],'EdgeColor','none')
    yline(weeklyRiskFree)
    legend("Portfolio Supplementary","Benchmark","Portfolio Main Text","Risk Free Rate")
    xlabel("Time (w)")
    ylabel("Performance")
    title("Performace")
end

% Quotes 
if hmmMethod==2
    % Quotes
    figure()
    hold on
    plot(quotesPort)
    plot(quotesBench)
    %---(OPTIONAL) Add S&P500 Quotes
    plot(quotesMarket)        
    legend("Portfolio Supplementary","Benchmark","S&P500","Location","southeast")
    % legend("Portfolio","Benchmark","Location","northwest")
    xlabel("Time (w)")
    ylabel("Quotes")
    title("Quotes")
else
    % Quotes comparison
    figure()
    hold on
    plot(quotesPort)
    plot(quotesBench)
    plot(quotesMarket)        
    plot(Temp.quotesPort)
    legend("Portfolio Supplementary","Benchmark","S&P500","Portfolio Main Text","Location","southeast")
    xlabel("Time (w)")
    ylabel("Quotes")
    title("Quotes")
end

% Annualized Volatilities
figure()
hold on
plot(volatPort)
plot(volatBench)
plot(volatActive, 'LineStyle','--')
plot(downsideStdDev, 'LineStyle','--')
legend("Portfolio","Benchmark","Track Err","Downside St Dev","Location","east")
xlabel("Time (w)")
ylabel("Volatily, annualized")
title("Volatility")

% Ratios
figure()
hold on
plot(infoRatio)
plot(sharpeRatio)
plot(sortinoRatio)
plot(treynorRatio)
legend("Information","Sharpe","Sortino","Treynor","Location","east")
xlabel("Time (w)")
ylabel("Ratios")
title("Ratios")

% Ratios comparison
if hmmMethod==1
    % Info Ratios
    figure()
    hold on
    plot(infoRatio)
    plot(Temp.infoRatio)
    legend("Portfolio Supplementary","Portfolio Main Text","Location","southeast")
    xlabel("Time (w)")
    ylabel("Ratio")
    title("Information Ratios")
    
    % Sharpe Ratios
    figure()
    hold on
    plot(sharpeRatio)
    plot(Temp.sharpeRatio)
    legend("Portfolio Supplementary","Portfolio Main Text","Location","southeast")
    xlabel("Time (w)")
    ylabel("Ratio")
    title("Sharpe Ratios")
    
    % Sortino Ratios
    figure()
    hold on
    plot(sortinoRatio)
    plot(Temp.sortinoRatio)
    legend("Portfolio Supplementary","Portfolio Main Text","Location","southeast")
    xlabel("Time (w)")
    ylabel("Ratio")
    title("Sortino Ratios")
    
    % Treynor Ratios
    figure()
    hold on
    plot(treynorRatio)
    plot(Temp.treynorRatio)
    legend("Portfolio Supplementary","Portfolio Main Text","Location","southeast")
    xlabel("Time (w)")
    ylabel("Ratio")
    title("Treynor Ratios")
end

%--- REPORT PORTFOLIO PERFORMANCES (PRINTS) ------------------------------

% Check occurrencies of estimated stress state
fprintf("\nOccurrencies of estimated stress state = %d\n",nnz(estStatesVec==1))

% Print time average of weights
fprintf("\nMean Weight of every asset: \n")
fprintf("%.4f\n",mean(wMatrix))

% Print Indicators at the last week (T=52)
fprintf("\n\t\t\t\tBenchmark\tPortfolio\n")
fprintf("Q_{52} \t\t\t& %.4f \t& %.4f\n",    quotesBench(end), quotesPort(end))
fprintf("Tot Ret \t\t& %.4f \t& %.4f\n",    totRetBench(end)*100, totRetPort(end)*100)
fprintf("Tot Active Ret \t&  -\t\t& %.4f\n", totRetActive(end)*100)
fprintf("Avg Ret \t\t& %.4f \t& %.4f\n",      avgRetBench(end)*100, avgRetPort(end)*100)
fprintf("Avg Active Ret \t&  -\t\t& %.4f\n", avgActiveRet(end)*100)
fprintf("Abs Risk \t\t& %.4f \t& %.4f\n",     volatBench(end)*100, volatPort(end)*100)
fprintf("Rel Risk \t\t&   -\t\t& %.4f\n\n",      volatActive(end)*100)

fprintf("Info Ratio \t\t& -\t\t& %.4f\n",      infoRatio(end))
fprintf("Sharpe Ratio \t& -\t\t& %.4f\n",    sharpeRatio(end))
fprintf("Sortino Ratio \t& -\t\t& %.4f\n",   sortinoRatio(end))
fprintf("Treynor Ratio \t& -\t\t& %.4f\n",   treynorRatio(end))















%%

% Helper Functions
%--------------------------------------------------------------------------

function [dates,data,returns,dataMarket,returnsMarket] = loadData()

    % select data columns to load (1=load, 0=ignore) 
    columnIndices = [6, 13, 20, 27, 90, 34, 44, 54, 64, 74, 94, 107, 113, 119, 123]; 

    % load from file
    load('GlobalDB.mat','IndicesDataW','ObsDatesW','IndicesDescr')
    % check loading correct columns
    fprintf("Loaded Columns:\n")
    for i=1:length(columnIndices)
        fprintf(" - %s\n", IndicesDescr{4,columnIndices(i)})
    end

    % assign data and dates
    data = IndicesDataW(:, columnIndices);
    dates    = ObsDatesW;

    % load market from file
    columnIndexMarket = 120;
    dataMarket = IndicesDataW(:, columnIndexMarket);
    fprintf("\nLoaded market: %s\n\n",IndicesDescr{4,columnIndexMarket});
    
    % Manage Nans: keep only most recent data without NaNs
    data(data==-999999999)=NaN;
    idxRowNan = max(find( any(isnan(data),2) ));
    data = data(idxRowNan+1:end,:);
    dates = dates(idxRowNan+1:end,:);
    dataMarket = dataMarket(idxRowNan+1:end,:);

    % Define returns 
    returns = (data(2:end,:)-data(1:end-1,:))./data(1:end-1,:);  
    returnsMarket = (dataMarket(2:end,:)-dataMarket(1:end-1,:))./dataMarket(1:end-1,:);  

end
%--------------------------------------------------------------------------

function [inliers, outliers,distVec] = CK_inoutliers(returns, alpha)

    % historical mean and covariance
    meanHist = mean(returns); % time mean for each asset
    covHist  = cov(returns);  % covariance matrix

    % squared Mahalanobi distance
    T = size(returns,1);
    distVec = zeros(T,1);
    for i=1:T
        distVec(i) = (returns(i,:) - meanHist) * inv(covHist) * (returns(i,:) - meanHist)'; % distance for every t
    end

    % inliers and outliers
    isOut = distVec > chi2inv(1-alpha,size(returns,2));
    outliers = returns( isOut,:);
    inliers  = returns(~isOut,:);
end
%--------------------------------------------------------------------------

function [estTr,estEm,estPrior,estStates,pStress] = distressHMM(seqContin,guessTr,guessEm,guessPrior)

    % model parameters
    numObservations = size(guessEm,2); 

    % Define the sequence of observations (bin the constinuous sequence, so that each bin is observed many times)
    edges = linspace(min(seqContin),max(seqContin),numObservations+1); % bins edges
    seqDiscr = discretize(seqContin, edges); %assign bin index to each element of seqContin

    % Estimate Hidden Markov Model (transition and emission probabilities)
    [estTr, estEm] = hmmtrain(seqDiscr', guessTr, guessEm);
    
    % Set extended guess Transition and Observation Matrices (to include priors on initial states)
    estTrExt = [0, guessPrior; zeros(2,1), estTr];
    estEmExt = [zeros(1,size(guessEm,2)); estEm];
    % Recover posterior probabilities of being in state 1 or 2, at every time
    pStates = hmmdecode(seqDiscr',estTrExt,estEmExt);
    % Drop extension
    pStates = pStates(2:end,:);

    % Update Prior of initial state (t=2, will be t=1 of next rolling window)
    estPrior = pStates(:,2)';

    % Define most probable state, at every time
    [~,estStates] = max(pStates,[],1);
    % (OPTIONAL) Compare Viterbi (does not account for prior)
    % estStates2 = hmmviterbi(seqDiscr',estTr,estEm);
    % fprintf("nr of differently estimated states: %d\n", nnz(estStates2-estStates));

    %--- Select a criterion to label the states (calm or stress)
    % Assume the stress state is the one with less occurrencies 
    countOccurrencies = histcounts(estStates);
    [~,stateStress] = min(countOccurrencies);
    % Assume the stress state is the one with higher distVec mean
    is1 = (estStates==1);
    [~,stateStress2] = max([mean(seqContin(is1)), mean(seqContin(~is1))]);
    %fprintf("mean state 1: %f state 2 : %f\n",mean(seqContin(is1)), mean(seqContin(~is1)))
    % Assume the stress state is the one with higher probty of emitting last bin of obs (more extreme values)
    [~,stateStress3] = max(estEm(:,end));
    % Assume the stress state is the one with lower probability of staying
    [~,stateStress4] = min(diag(estTr));
    % (OPTIONAL) Compare
    fprintf("Compare stress state with four methods: %i-%i-%i-%i\n",stateStress,stateStress2,stateStress3,stateStress4)

    % Recover (most probable) state at current time
    pStress = pStates(:,end)' *  estTr(:,stateStress4);

end

function [A, mu, sigma, p, smoothed] = fithmm(y)
    
    % SCOPE: This Program Estimates (via Maximum Likelihood under Normality Hypothesis, and by Using the Baum–Welch Algorithm) the Parameters of a Hidden Markov Model, 
    %        i.e. Estimates the Dynamic Probability of Different States.
    %
    % -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    %
    % USE: [A, mu, sigma, p, smoothed] = fithmm(y);
    %
    % -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    %
    % INPUT:
    %
    % y = Numerical Array (Nobs-by-1) containing the Sequence of States, i.e. Market Conditions (Normal = 1, Distressed = 2).
    %
    % -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    %
    % Written on October 23rd, 2014 by:
    %
    % MICHELE TROVA
    % Head of Analysis & Financial Controlling
    % Area Finance - Group Finance Dept.
    % VENETO BANCA S.C.p.A.
    % Via Feltrina Sud, 250
    % 31044 Montebelluna (TV)
    % Italy
    % Tel.      +0039-0423-283757
    % Fax.      +0039-0423-283340
    % e-mail1:  michele.trova@venetobanca.it
    % e-mail2:  anfincont@venetobanca.it
    % web site: http://www.venetobanca.it
    %
    % based on:
    %
    % "Regime Shifts: Implications for Dynamic Strategies"
    % by Mark Kritzman, Sébastien Page, and David Turkington
    % Financial Analysts Journal
    % Volume 68, Number 3, 2012
    % Appendix B. MATLAB Code for Estimating Regimes
    % pag. 36-37
    %
    % -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    % Compute the Numer of Observations in the Sample and Define Simple Initial Guesses for the Model Parameters (Can Be Changed) ---------------------------------------------
    
    T = size(y, 1);                                                             % Number of Observations
    
    mu                   = [mean(y), mean(y)] + randn(1, 2) * std(y);           % Parameters Settings
    sigma                = [std(y), std(y)];
    A                    = [.8, .2; .2, .8];
    p                    = 0.50;
    iteration            = 2;
    likelihood(1)        = -999.000;
    change_likelihood(1) = Inf;
    tolerance            = 0.000001;
    
    % -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    
    
    
    % Estimate the Model Parameters -------------------------------------------------------------------------------------------------------------------------------------------
    
    while change_likelihood(iteration-1) > tolerance
        
        for t = 1:T                                                            % Step 0 - Compute the Probability of Observing Data, Based on Gaussian PDF
            B(t, 1) = exp(-.5 * ((y(t) - mu(1)) / sigma(1)) .^ 2) / (sqrt(2 * pi) * sigma(1));
            B(t, 2) = exp(-.5 * ((y(t) - mu(2)) / sigma(2)) .^ 2) / (sqrt(2 * pi) * sigma(2));
        end
        forward(1, :) = p .* B(1, :);
        scale(1, :)   = sum(forward(1, :));
        forward(1, :) = forward(1, :) / sum(forward(1, :));
        
        for t = 2:T                                                           % Step 1 - Compute the Probability of the Regimes Given Past Data
            forward(t, :) = (forward(t-1, :) * A) .* B(t, :);
            scale(t, :)   = sum(forward(t, :));
            forward(t, :) = forward(t, :) / sum(forward(t, :));
        end
        backward(T, :) = B(T, :);
        backward(T, :) = backward(T, :) / sum(backward(T, :));
        
        for t = T-1:-1:1                                                       % Step 2 - Compute the Probability of the Regimes Given Future Data
            backward(t, :) = (A * backward(t+1, :)')' .* B(t+1, :);
            backward(t, :) = backward(t, :) / sum(backward(t, :));
        end
        
        for t = 1:T                                                           % Steps 3 & 4 - Compute the Probability of the Regimes Given All Data
            smoothed(t, :) = forward(t, :) .* backward(t, :);
            smoothed(t, :) = smoothed(t, :) / sum(smoothed(t, :));
        end
        
        for t = 1:T-1                                                        % Step 5 - Compute the Probability of Each Transition Having Occurred
            xi(:, :, t) = (A .* (forward(t, :)' * (backward(t+1, :) .* B(t+1, :))));
            xi(:, :, t) = xi(:, :, t) / sum(sum(xi(:, :, t)));
        end
        p                            = smoothed(1, :);
        exp_num_transitions          = sum(xi, 3);
        A(1, :)                      = exp_num_transitions(1, :) / sum(sum(xi(1, :, :), 2), 3);
        A(2, :)                      = exp_num_transitions(2, :) / sum(sum(xi(2, :, :), 2), 3);
        mu(1)                        = (smoothed(:, 1)' * y)' / sum(smoothed(:, 1));
        mu(2)                        = (smoothed(:, 2)' * y)' / sum(smoothed(:, 2));
        sigma(1)                     = sqrt(sum(smoothed(:, 1) .* (y - mu(1)) .^ 2) / sum(smoothed(:, 1)));
        sigma(2)                     = sqrt(sum(smoothed(:, 2) .* (y - mu(2)) .^ 2) / sum(smoothed(:, 2)));
        likelihood(iteration+1)      = sum(sum(log(scale)));
        change_likelihood(iteration) = abs(likelihood(iteration+1) - likelihood(iteration));
        iteration                    = iteration + 1;
    
    end
    
    % -------------------------------------------------------------------------------------------------------------------------------------------------------------------------


end



