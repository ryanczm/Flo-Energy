# Power Trading Firm - Technical Assignment (Trading & Portfolio Management Team)

This repo is my solution to the take home technical assignment for a retail power trading company (buy in wholesale market, sell in retail market) for a __power trading analyst__ role. I was informed, upon rejection, that one day before my presentation to the company (of this assignment), a senior member of the team had requested a lateral transfer from a software engineer role into this role. Thus, there was no more headcount for the role. This was despite 3 rounds of interviews and also despite the fact that my solution was the most well-answered out of all the applicants and they were very happy with my technical/data analysis skills in SQL and Python.

I am thus posting this for any power-trading related roles. The answers and code are in `answers.ipynb`.

# Introduction

As discussed during your first interview, the Trading and Portfolio Management team of (redacted) is mostly involved with energy pricing and purchasing, so it makes sense to base our technical assessment on these two elements. You will be working with 3 small datasets with spot prices, futures settlements and consumption data, these will typically be your starting point for many analyses within (redacted):


* _Spot prices_ are the half hourly realized market prices (SGD per MWh) for electricity in Singapore. This market is operated by the EMC and creates a price for each half hour for generators to sell their energy and end consumers or retailers to buy their electricity demand for that particular half hour period. The prices are capped at a maximum of 4500 SGD per MWh, negative prices for oversupply rarely happen because of the low generation share of renewable energy. There are quite a few safety mechanisms in place (e.g. demand response) to prevent these price spikes to the maximum from happening, but they do occur from time to time and tend to cluster. They are difficult to predict by nature, usually the result of a combination of high demand resulting from weather circumstances and undersupply due to maintenance or outages, and can have a large impact on Flo’s profitability.
* _Futures price_ settlements typically represent the latest market consensus outlook (SGD per MWh) on what the average of spot prices is going to be between a particular start and end date (e.g. Q1 '24). Flo uses these futures prices to offer customers a contract price for a certain period. In Singapore these contracts usually trade for a range of quarters, but the underlying month averages may vary quite a bit due to e.g. weather conditions and the number of public holidays.
● Consumption data are typically received in half hourly kWh values. Reminder: 1 MWh = 1000 kWh.

# Questions

1. Treating the SpotPrices_EMC CSV like it is a table named spotprices, write a SQL query to analyse the average, min and max price and demand values per year-month combination for 2022 and 2023

---

2. Now use the historical half hourly price patterns to create a normalized half hourly price pattern per half hour per weekday per month. You will be asked later to apply this on quarterly futures settlement price data. Write down your (data and portfolio management strategic) assumptions for this half hourly price curve. Also write out how you would query this in SQL.

| Month | Weekday | HalfHour | Normalized Pricing Pattern |
|-------|---------|----------|-----------------------------|
| 1     | 1       | 1        | x%                          |

---

3. The consumption data contains a hotel, a data centre, a school and a church. Correctly label the connection ids with their industry type and write down your reasoning for why which one is which.

---

4. For each of these, create a normalized half hourly consumption pattern per half hour per weekday based on their available historical data. For simplicity you don’t have to consider seasonality effects (e.g. more demand with higher temperature) and the effect of public holidays (less consumption during days off). Expected output would look something like this

| Weekday | HalfHour | Normalized Consumption Pattern |
|---------|----------|-----------------------------|
| 1       | 1        | x%                          |

---

5. The settlement prices Settlements_SGX CSV shows the market price expectations for 2024-2026.

* Use your normalized half hourly price curve and your half hourly consumption curve on the settlement prices and consumption volumes to create half hourly expected market prices (SGD/MWh) and half hourly expected consumption volumes (MWh) per connection
between 01-01-2024 and 31-12-2026. 

* Combine these to arrive at each connection’s monthly expected energy costs between 01-01-2024 and 31-12-2026 for each of these customers.
* Calculate the assumed costs per MWh for each connection vs the average settlement monthly price, and the difference value (delta) between the two. What does a high delta indicate?

---

6. Assuming a 2-year contract for the hotel, and 3-year contracts for the remaining connections starting 1-1-2024, calculate how much energy we would have to buy in total per month for this small retail portfolio.