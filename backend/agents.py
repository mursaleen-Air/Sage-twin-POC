# agents.py

def revenue_agent(state, price_increase, marketing_increase):
    revenue = state["revenue"]

    # Simple rule logic
    revenue *= (1 + price_increase / 100 * 0.8)
    revenue *= (1 + marketing_increase / 100 * 0.5)

    return round(revenue, 2)


def sentiment_agent(state, price_increase, added_delay):
    sentiment = state["sentiment"]

    sentiment -= price_increase * 0.3
    sentiment -= added_delay * 5

    return max(0, min(100, round(sentiment, 2)))


def risk_agent(sentiment, delivery_delay):
    risk = 100 - sentiment + delivery_delay * 5
    return max(0, min(100, round(risk, 2)))


def strategy_agent(revenue, sentiment, risk):
    if risk > 70:
        return "High operational risk detected. Reduce delays and stabilize pricing."
    elif sentiment < 50:
        return "Customer sentiment declining. Improve delivery experience."
    elif revenue > 120000:
        return "Growth strategy working. Consider scaling marketing further."
    else:
        return "Business stable. Monitor KPIs closely."
