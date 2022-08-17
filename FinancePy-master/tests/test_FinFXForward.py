###############################################################################
# Copyright (C) 2018, 2019, 2020 Dominic O'Kane
###############################################################################

from financepy.utils.date import Date
from financepy.products.rates.ibor_deposit import IborDeposit
from financepy.products.rates.ibor_single_curve import IborSingleCurve
from financepy.utils.calendar import CalendarTypes
from financepy.utils.day_count import DayCountTypes
from financepy.products.fx.fx_forward import FXForward


def test_FinFXForward():
    #  https://stackoverflow.com/questions/48778712
    #  /fx-vanilla-call-price-in-quantlib-doesnt-match-bloomberg

    valuation_date = Date(13, 2, 2018)
    expiry_date = valuation_date.add_months(12)
    # Forward is on EURUSD which is expressed as number of USD per EUR
    # ccy1 = EUR and ccy2 = USD
    forName = "EUR"
    domName = "USD"
    currency_pair = forName + domName  # Always ccy1ccy2
    spot_fx_rate = 1.300  # USD per EUR
    strike_fx_rate = 1.365  # USD per EUR
    ccy1InterestRate = 0.02  # USD Rates
    ccy2InterestRate = 0.05  # EUR rates

    spot_days = 0
    settlement_date = valuation_date.add_weekdays(spot_days)
    maturity_date = settlement_date.add_months(12)
    notional = 100.0
    calendar_type = CalendarTypes.TARGET

    depos = []
    fras = []
    swaps = []
    deposit_rate = ccy1InterestRate
    depo = IborDeposit(settlement_date, maturity_date, deposit_rate,
                       DayCountTypes.ACT_360, notional, calendar_type)
    depos.append(depo)
    for_discount_curve = IborSingleCurve(valuation_date, depos, fras, swaps)

    depos = []
    fras = []
    swaps = []
    deposit_rate = ccy2InterestRate
    depo = IborDeposit(settlement_date, maturity_date, deposit_rate,
                       DayCountTypes.ACT_360, notional, calendar_type)
    depos.append(depo)
    dom_discount_curve = IborSingleCurve(valuation_date, depos, fras, swaps)

    notional = 100.0
    notional_currency = forName

    fxForward = FXForward(expiry_date,
                          strike_fx_rate,
                          currency_pair,
                          notional,
                          notional_currency)

    fwdValue = fxForward.value(valuation_date, spot_fx_rate,
                               dom_discount_curve, for_discount_curve)

    fwdFXRate = fxForward.forward(valuation_date, spot_fx_rate,
                                  dom_discount_curve,
                                  for_discount_curve)

    assert round(fwdFXRate, 4) == 1.3388

    assert round(fwdValue['value'], 4) == -2.4978
    assert round(fwdValue['cash_dom'], 4) == -249.7797
    assert round(fwdValue['cash_for'], 4) == -192.1382
    assert fwdValue['not_dom'] == 136.5
    assert fwdValue['not_for'] == 100.0
    assert fwdValue['ccy_dom'] == 'USD'
    assert fwdValue['ccy_for'] == 'EUR'
