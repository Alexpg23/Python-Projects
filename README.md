                                   ############### INDEX ############### 

This repository contains several Python, R and excel projects created by me individually or as part of a team
along my academic and professional preparation. The notebooks contain several example of finance-related
problem solving related to Portfolio Management, Risk Management, Derivatives Pricing, Investment Strategies
and Machine Leargning Applications. This Index is aimed to help readers to find the relevant files among all 
the support files and to give a general insight of the problem solved and models used within each project. All
of the projects referenced below are developed in Python or R.



1. BVAR with Stochastic Vol - Thesis


  The change in stock prices dynamic during recent years have made impossible to disregard the
    role that the so-called "democratization" of financial markets have played in the augmented
    volatility experienced in it. In this sense, it has become more important than ever to correctly
    identify the risks that investors are adding to their portfolio. I proposed to perform variance
    decomposition based on a Vector Autoregressive model and a log-linearization specification of a
    time-varying discounted dividend asset pricing model. This analysis is focused mainly on the
    S&P 500 but it is easily replicable at the firm level. This will allow us to compute a speculation
    ratio, whose evolution through time can help us to determine when stock movements are being
    motivated by speculation reasons rather than fundamental ones. I also show that this metric can
    serve as an indicator to implement a hedging strategy to enhance the risk-adjusted performance
    of a long-term investor.
   1.1  BVAR with Stochastic Vol - Thesis.ipynb
   1.2  BVAR.Rmd

  
2. MSc Financial Engineering


  In this folder, you can find all the projects realized during my master program at EDHEC Business
    School. There are examples of Advanced Derivatives models, Risk Management models, Machine 
    Learning Applications to finance, Investment Strategies using Machine Learning Methods, etc.
_    
    2.1 Derivatives
      In this folder, there are several excersices applying advanced derivatives concepts in Python 
        like Delta Hedging Strategies (with or without costs), Stock Price distribution implied in
        Volatility Smile, One Touch option pricing, CDS model calibration, Swaption pricing, etc.
        2.1.1  Advanced_Derivatives_Coursework_final.ipynb
        2.1.2  Advanced_Derivatives_One_Touch.ipynb
        2.1.3  CDS.xlsm
        2.1.4  Delta Hedging Hull.xlsm
        2.1.5  Swap Curve.xlsm
_        
    2.2 FinancyPy-Master
      In this folder, you can find pre-built functions, documentation and example of several derivatives
        instruments, for both vanilla and more exotic ones, contained in the FinancePy package that was 
        created by a Dominic O'Kane current professor at EDHEC Business School and former Head of the
        Quantitative Research team at Lehman Brothers
_
    2.3 Graphviz
      This is a pre-built package facilitates the creation and rendering of graph descriptions in the 
        DOT language of the Graphviz graph drawing software (upstream repo) from Python.
_
    2.4 Introduction to Machine Learning
      In this folder, you will find several examples of Machine Learning Models focused on Supervised 
        Learnign such as Logistic Regression, K-Nearest Neighbor, Decision Trees, Support Vector Machine
        Lasso and Ridge Regressions, AdaBoost Classifier, Random Forest, etc. 
        2.4.1 IntroToML_Coursework_One
          2.4.1.1  ML_Coursework_1_79083_79794_73713.ipynb
        2.4.2  IntroToML_Coursework_Two
          2.4.2.1  ML_Coursework_2_79083_79794_73713.ipynb
 _
    2.5 Machine Learning Applications
      In this folder, there additional excersices introducing different Manchine Learning, now exploring 
        Unsupervised Learnign and Reinforcement Learning, some of which are applied into finance problems.
        Some of the models that are used are Single and Multilayer Perceptons regressions, Artificial Neural 
        Networks (applied to options pricing and hedging), Natural Language Processing (Bag of Words, Words
        Embedding, etc.), Autoencoders, Recursive Neural Networks, Sequence-To-Sequence models, Reinforcement
        Learnign model, etc.
        2.5.1  Coursework
          2.5.1.1  ML_IN_FINANCE_COURSEWORK_79803_73713_79793
            2.5.1.1.1  ML_IN_F_COURSEWORK_79803_73713_79793.ipynb
_
    2.6 Market Microstructure
      In this folder, you can find a liquidity-based investment strategy implemented in Python. The strategy
        is based on Bid-Ask spreand and Volume analysis.
        2.6.1  Liquidity Trading Strategy.ipynb
_
    2.7 Risk Measurement
      In this folder, there is some excersices implementing different Risk Management strategies. Some of them 
        are Volatility fitting through GARCH and EWMA models, Valut-at-Risk and Expected Shortfall estimation 
        through parametric and non-parametric methods, and accuracy scoring of both volatility and risk 
        measures estimates. Risk attribution, Stress Testing and Backtesting are implemented in order to correctly
        specify sources of risk and to provide insights about the risk profile of an investment portfolio.
        2.7.1  RM_Coursework_1.ipynb
        2.7.2  RM_Coursework_2.ipynb
_
    2.8 Sequential Investing
      In this folder, you can find an investment strategy that was proposed and developed by myself using Machine
        Learnign models in Python. The strategy is a hedging strategy using a market volatility proxy like VIX. The
        model tries to extract signlas from the markets to predcit when would it be necessary to hedge the market
        exposure and in what proportion.The strategy succeds in outperforming the market.
        2.8.1  Prediciton and Sequential Investment Strategies-Final-Assignment-73713.ipynb
-
    2.9 VBA
      In this folder, there is some derivatives and finance-realated excersices soved using Visual Basic. Some of them 
        are the pricing of a forward starting option and a Vascisek model calibration using Monte Carlo simulation.
        2.9.1  PALACIOS_Alejandro_SANTOSDECARVALHO_Marcel_VBA_FA.xlsm
