# multi_agents.py
"""
Step 4: Multi-Agent System
Each agent is responsible for a specific domain and produces transparent outputs.
"""

from typing import Dict, Any, List
from datetime import datetime
from causal_graph import propagate_change


class AgentOutput:
    """Standardized output from each agent"""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.timestamp = datetime.now().isoformat()
        self.changes: Dict[str, float] = {}
        self.rules_triggered: List[str] = []
        self.reasoning: str = ""
        self.confidence: float = 0.0
        self.warnings: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "timestamp": self.timestamp,
            "changes": self.changes,
            "rules_triggered": self.rules_triggered,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "warnings": self.warnings
        }


class RevenueAgent:
    """
    Agent 1: Revenue Agent
    Adjusts revenue based on: Customers, Pricing, Marketing
    """
    name = "Revenue Agent"
    
    def process(self, state: Dict[str, float], adjustments: Dict[str, float]) -> AgentOutput:
        output = AgentOutput(self.name)
        
        current_revenue = state.get("revenue", 100000)
        new_revenue = current_revenue
        
        # Rule 1: Customer impact on revenue
        customer_change = adjustments.get("customers", 0)
        if customer_change != 0:
            revenue_from_customers = current_revenue * (customer_change / 100) * 0.7
            new_revenue += revenue_from_customers
            output.rules_triggered.append(f"Customer change ({customer_change:+.1f}%) → Revenue {revenue_from_customers:+,.0f}")
        
        # Rule 2: Price impact on revenue
        price_change = adjustments.get("price", 0)
        if price_change != 0:
            # Price increase: revenue up but may lose customers
            revenue_from_price = current_revenue * (price_change / 100) * 0.5
            new_revenue += revenue_from_price
            output.rules_triggered.append(f"Price change ({price_change:+.1f}%) → Revenue {revenue_from_price:+,.0f}")
            
            if price_change > 10:
                output.warnings.append("High price increase may increase churn")
        
        # Rule 3: Marketing impact on revenue  
        marketing_change = adjustments.get("marketing_spend", 0)
        if marketing_change != 0:
            # Marketing increases revenue via customer acquisition
            revenue_from_marketing = current_revenue * (marketing_change / 100) * 0.3
            new_revenue += revenue_from_marketing
            output.rules_triggered.append(f"Marketing change ({marketing_change:+.1f}%) → Revenue {revenue_from_marketing:+,.0f}")
        
        # Calculate final change
        revenue_change_pct = ((new_revenue - current_revenue) / current_revenue) * 100 if current_revenue > 0 else 0
        
        output.changes = {
            "revenue": round(new_revenue, 2),
            "revenue_change_pct": round(revenue_change_pct, 2)
        }
        
        output.reasoning = f"Revenue adjusted from ${current_revenue:,.0f} to ${new_revenue:,.0f} based on {len(output.rules_triggered)} factors"
        output.confidence = min(95, 70 + len(output.rules_triggered) * 5)
        
        return output


class CustomerAgent:
    """
    Agent 2: Customer Agent
    Adjusts: Customer count, Churn, Retention
    """
    name = "Customer Agent"
    
    def process(self, state: Dict[str, float], adjustments: Dict[str, float]) -> AgentOutput:
        output = AgentOutput(self.name)
        
        current_customers = state.get("customers", 1000)
        current_churn = state.get("churn_rate", 5)
        
        new_customers = current_customers
        new_churn = current_churn
        
        # Rule 1: Marketing drives acquisition
        marketing_change = adjustments.get("marketing_spend", 0)
        if marketing_change != 0:
            customer_gain = current_customers * (marketing_change / 100) * 0.4
            new_customers += customer_gain
            output.rules_triggered.append(f"Marketing ({marketing_change:+.1f}%) → {customer_gain:+,.0f} customers")
        
        # Rule 2: Price affects customer retention
        price_change = adjustments.get("price", 0)
        if price_change != 0:
            customer_loss = current_customers * (price_change / 100) * 0.15
            new_customers -= customer_loss
            new_churn += price_change * 0.1
            output.rules_triggered.append(f"Price ({price_change:+.1f}%) → Churn impact")
        
        # Rule 3: Sentiment affects retention
        sentiment_change = adjustments.get("sentiment", 0)
        if sentiment_change != 0:
            retention_impact = current_customers * (sentiment_change / 100) * 0.2
            new_customers += retention_impact
            new_churn -= sentiment_change * 0.05
            output.rules_triggered.append(f"Sentiment ({sentiment_change:+.1f}%) → Retention impact")
        
        # Rule 4: Delivery delay increases churn
        delay_change = adjustments.get("delivery_delay", 0)
        if delay_change != 0:
            new_churn += delay_change * 0.5
            churn_impact = current_customers * (delay_change * 0.5 / 100)
            new_customers -= churn_impact
            output.rules_triggered.append(f"Delay ({delay_change:+.1f} days) → Churn +{delay_change * 0.5:.1f}%")
        
        # Clamp values
        new_customers = max(0, new_customers)
        new_churn = max(0, min(100, new_churn))
        
        output.changes = {
            "customers": round(new_customers),
            "churn_rate": round(new_churn, 2),
            "customer_change_pct": round(((new_customers - current_customers) / current_customers) * 100, 2) if current_customers > 0 else 0
        }
        
        output.reasoning = f"Customer base adjusted to {new_customers:,.0f}, churn rate to {new_churn:.1f}%"
        output.confidence = min(90, 65 + len(output.rules_triggered) * 6)
        
        if new_churn > 10:
            output.warnings.append("Churn rate is concerning (>10%)")
        
        return output


class SentimentAgent:
    """
    Agent 3: Sentiment Agent
    Adjusts: Sentiment score, Brand health
    """
    name = "Sentiment Agent"
    
    def process(self, state: Dict[str, float], adjustments: Dict[str, float]) -> AgentOutput:
        output = AgentOutput(self.name)
        
        current_sentiment = state.get("sentiment", 75)
        new_sentiment = current_sentiment
        
        # Rule 1: Price increases hurt sentiment
        price_change = adjustments.get("price", 0)
        if price_change != 0:
            sentiment_impact = -price_change * 0.3
            new_sentiment += sentiment_impact
            output.rules_triggered.append(f"Price ({price_change:+.1f}%) → Sentiment {sentiment_impact:+.1f}")
        
        # Rule 2: Delivery delays hurt sentiment significantly
        delay_change = adjustments.get("delivery_delay", 0)
        if delay_change != 0:
            sentiment_impact = -delay_change * 5  # Each day of delay = -5 sentiment
            new_sentiment += sentiment_impact
            output.rules_triggered.append(f"Delay ({delay_change:+.1f} days) → Sentiment {sentiment_impact:+.1f}")
            
            if delay_change > 2:
                output.warnings.append("Significant delivery delays will harm brand perception")
        
        # Rule 3: Marketing can improve sentiment
        marketing_change = adjustments.get("marketing_spend", 0)
        if marketing_change > 0:
            sentiment_boost = marketing_change * 0.1
            new_sentiment += sentiment_boost
            output.rules_triggered.append(f"Marketing → Sentiment +{sentiment_boost:.1f}")
        
        # Rule 4: Customer satisfaction correlation
        satisfaction_change = adjustments.get("customer_satisfaction", 0)
        if satisfaction_change != 0:
            new_sentiment += satisfaction_change * 0.8
            output.rules_triggered.append(f"Satisfaction → Sentiment")
        
        # Clamp 0-100
        new_sentiment = max(0, min(100, new_sentiment))
        
        output.changes = {
            "sentiment": round(new_sentiment, 1),
            "sentiment_change": round(new_sentiment - current_sentiment, 1),
            "brand_health": "good" if new_sentiment >= 70 else "concerning" if new_sentiment >= 50 else "critical"
        }
        
        output.reasoning = f"Sentiment adjusted from {current_sentiment:.0f} to {new_sentiment:.0f}/100"
        output.confidence = 80
        
        if new_sentiment < 50:
            output.warnings.append("Low sentiment threatens customer retention")
        
        return output


class OperationsAgent:
    """
    Agent 4: Operations Agent
    Adjusts: Delivery delay, Operational efficiency
    """
    name = "Operations Agent"
    
    def process(self, state: Dict[str, float], adjustments: Dict[str, float]) -> AgentOutput:
        output = AgentOutput(self.name)
        
        current_delay = state.get("delivery_delay", 2)
        current_costs = state.get("costs", 50000)
        
        new_delay = current_delay + adjustments.get("delivery_delay", 0)
        new_costs = current_costs
        
        # Rule 1: Cost cutting may increase delays
        cost_change = adjustments.get("costs", 0)
        if cost_change < 0:  # Cutting costs
            delay_impact = abs(cost_change) * 0.05  # 20% cost cut = 1 day more delay
            new_delay += delay_impact
            new_costs = current_costs * (1 + cost_change / 100)
            output.rules_triggered.append(f"Cost cut ({cost_change:.1f}%) → Delay +{delay_impact:.1f} days")
        elif cost_change > 0:  # Investing more
            delay_reduction = cost_change * 0.03
            new_delay = max(0, new_delay - delay_reduction)
            new_costs = current_costs * (1 + cost_change / 100)
            output.rules_triggered.append(f"Cost increase ({cost_change:+.1f}%) → Delay -{delay_reduction:.1f} days")
        
        # Calculate efficiency score
        efficiency = max(0, min(100, 100 - (new_delay * 10)))
        
        output.changes = {
            "delivery_delay": round(max(0, new_delay), 1),
            "costs": round(new_costs, 2),
            "operational_efficiency": round(efficiency, 1)
        }
        
        output.reasoning = f"Delivery delay: {new_delay:.1f} days, Efficiency: {efficiency:.0f}%"
        output.confidence = 85
        
        if new_delay > 5:
            output.warnings.append("Delivery delays exceeding 5 days will significantly impact customer satisfaction")
        
        return output


class RiskAgent:
    """
    Agent 5: Risk Agent
    Calculates: Financial risk, Operational risk, Overall risk score
    """
    name = "Risk Agent"
    
    def process(self, state: Dict[str, float], agent_outputs: Dict[str, AgentOutput]) -> AgentOutput:
        output = AgentOutput(self.name)
        
        risk_factors = []
        
        # Factor 1: Revenue health
        revenue = state.get("revenue", 100000)
        baseline_revenue = 100000
        revenue_risk = max(0, (1 - revenue / baseline_revenue) * 30) if baseline_revenue > 0 else 15
        risk_factors.append(("revenue_risk", revenue_risk))
        output.rules_triggered.append(f"Revenue factor → Risk contribution: {revenue_risk:.1f}")
        
        # Factor 2: Sentiment risk
        sentiment = state.get("sentiment", 75)
        sentiment_risk = max(0, (100 - sentiment) * 0.3)
        risk_factors.append(("sentiment_risk", sentiment_risk))
        output.rules_triggered.append(f"Sentiment ({sentiment:.0f}) → Risk contribution: {sentiment_risk:.1f}")
        
        # Factor 3: Churn risk
        churn = state.get("churn_rate", 5)
        churn_risk = churn * 3
        risk_factors.append(("churn_risk", churn_risk))
        output.rules_triggered.append(f"Churn ({churn:.1f}%) → Risk contribution: {churn_risk:.1f}")
        
        # Factor 4: Operational risk (delivery)
        delay = state.get("delivery_delay", 2)
        operational_risk = delay * 5
        risk_factors.append(("operational_risk", operational_risk))
        output.rules_triggered.append(f"Delay ({delay:.1f} days) → Risk contribution: {operational_risk:.1f}")
        
        # Factor 5: Cost risk
        costs = state.get("costs", 50000)
        cost_ratio = (costs / revenue) * 100 if revenue > 0 else 50
        cost_risk = max(0, cost_ratio - 40)  # Risk increases if costs > 40% of revenue
        risk_factors.append(("cost_risk", cost_risk))
        
        # Calculate overall risk
        overall_risk = sum(r for _, r in risk_factors)
        overall_risk = max(0, min(100, overall_risk))
        
        # Determine risk level
        if overall_risk > 70:
            risk_level = "critical"
        elif overall_risk > 50:
            risk_level = "high"
        elif overall_risk > 30:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        output.changes = {
            "risk_score": round(overall_risk, 1),
            "risk_level": risk_level,
            "financial_risk": round(revenue_risk + cost_risk, 1),
            "operational_risk": round(operational_risk + churn_risk, 1),
            "brand_risk": round(sentiment_risk, 1)
        }
        
        output.reasoning = f"Overall risk: {overall_risk:.0f}/100 ({risk_level})"
        output.confidence = 90
        
        if overall_risk > 50:
            output.warnings.append(f"Risk level is {risk_level} - immediate attention required")
        
        return output


class StrategyAgent:
    """
    Agent 6: Strategy Agent (Executive Brain)
    Looks at all outputs and produces recommendations, warnings, tradeoff insights
    """
    name = "Strategy Agent"
    
    def process(
        self, 
        state: Dict[str, float], 
        agent_outputs: Dict[str, AgentOutput],
        adjustments: Dict[str, float]
    ) -> AgentOutput:
        output = AgentOutput(self.name)
        
        recommendations = []
        warnings = []
        tradeoffs = []
        
        # Analyze agent outputs
        risk_output = agent_outputs.get("Risk Agent")
        sentiment_output = agent_outputs.get("Sentiment Agent")
        customer_output = agent_outputs.get("Customer Agent")
        revenue_output = agent_outputs.get("Revenue Agent")
        
        # Collect all warnings from agents
        for agent_name, agent_output in agent_outputs.items():
            warnings.extend(agent_output.warnings)
        
        # Strategic analysis
        risk_score = state.get("risk_score", 50)
        sentiment = state.get("sentiment", 75)
        revenue = state.get("revenue", 100000)
        churn = state.get("churn_rate", 5)
        
        # Rule 1: High risk
        if risk_score > 60:
            recommendations.append("Reduce operational risk by optimizing delivery times")
            recommendations.append("Focus on cost efficiency without sacrificing quality")
            output.rules_triggered.append("High risk → Stabilization strategy")
        
        # Rule 2: Low sentiment
        if sentiment < 60:
            recommendations.append("Prioritize customer experience improvements")
            recommendations.append("Consider loyalty programs to retain at-risk customers")
            output.rules_triggered.append("Low sentiment → Customer focus strategy")
        
        # Rule 3: High churn
        if churn > 8:
            recommendations.append("Urgent: Implement churn reduction initiatives")
            recommendations.append("Analyze exit surveys and address top complaints")
            output.rules_triggered.append("High churn → Retention strategy")
        
        # Rule 4: Growth opportunity
        if risk_score < 30 and sentiment > 70:
            recommendations.append("Favorable conditions for growth investment")
            recommendations.append("Consider expanding marketing spend")
            output.rules_triggered.append("Low risk + high sentiment → Growth strategy")
        
        # Identify tradeoffs
        if adjustments.get("price", 0) > 0:
            tradeoffs.append("Price increase: Higher revenue potential vs. customer retention risk")
        
        if adjustments.get("costs", 0) < 0:
            tradeoffs.append("Cost reduction: Better margins vs. potential operational quality decline")
        
        if adjustments.get("marketing_spend", 0) > 0:
            tradeoffs.append("Marketing increase: Customer acquisition vs. short-term profitability")
        
        output.changes = {
            "recommendations": recommendations[:5],  # Top 5
            "warnings": warnings,
            "tradeoffs": tradeoffs,
            "strategic_priority": self._determine_priority(risk_score, sentiment, churn)
        }
        
        output.reasoning = f"Generated {len(recommendations)} recommendations based on {len(output.rules_triggered)} strategic rules"
        output.confidence = min(95, 75 + len(output.rules_triggered) * 5)
        output.warnings = warnings
        
        return output
    
    def _determine_priority(self, risk: float, sentiment: float, churn: float) -> str:
        if risk > 70:
            return "RISK MITIGATION"
        elif churn > 10:
            return "CUSTOMER RETENTION"
        elif sentiment < 50:
            return "BRAND RECOVERY"
        elif risk < 30 and sentiment > 70:
            return "GROWTH"
        else:
            return "OPTIMIZATION"


# ============================================
# MULTI-AGENT ORCHESTRATOR
# ============================================

class MultiAgentEngine:
    """
    Orchestrates all agents and manages the simulation pipeline
    """
    
    def __init__(self):
        self.revenue_agent = RevenueAgent()
        self.customer_agent = CustomerAgent()
        self.sentiment_agent = SentimentAgent()
        self.operations_agent = OperationsAgent()
        self.risk_agent = RiskAgent()
        self.strategy_agent = StrategyAgent()
    
    def run_simulation(
        self, 
        current_state: Dict[str, float], 
        adjustments: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Run full simulation with all agents.
        Returns comprehensive output with transparency.
        """
        agent_outputs: Dict[str, AgentOutput] = {}
        
        # Store baseline
        baseline_state = current_state.copy()
        working_state = current_state.copy()
        
        # Phase 1: Revenue, Customer, Sentiment, Operations (parallel domain agents)
        revenue_output = self.revenue_agent.process(working_state, adjustments)
        agent_outputs["Revenue Agent"] = revenue_output
        
        customer_output = self.customer_agent.process(working_state, adjustments)
        agent_outputs["Customer Agent"] = customer_output
        
        sentiment_output = self.sentiment_agent.process(working_state, adjustments)
        agent_outputs["Sentiment Agent"] = sentiment_output
        
        operations_output = self.operations_agent.process(working_state, adjustments)
        agent_outputs["Operations Agent"] = operations_output
        
        # Apply changes to working state
        for output in [revenue_output, customer_output, sentiment_output, operations_output]:
            for key, value in output.changes.items():
                if key in working_state and isinstance(value, (int, float)):
                    working_state[key] = value
        
        # Phase 2: Risk Agent (needs updated state)
        risk_output = self.risk_agent.process(working_state, agent_outputs)
        agent_outputs["Risk Agent"] = risk_output
        working_state["risk_score"] = risk_output.changes.get("risk_score", 50)
        
        # Phase 3: Strategy Agent (executive summary)
        strategy_output = self.strategy_agent.process(working_state, agent_outputs, adjustments)
        agent_outputs["Strategy Agent"] = strategy_output
        
        # Calculate Business Health Index
        health_score = self._calculate_health(working_state)
        
        # Build transparency report
        transparency_report = [output.to_dict() for output in agent_outputs.values()]
        
        return {
            "baseline_state": baseline_state,
            "projected_state": working_state,
            "adjustments": adjustments,
            "health_score": health_score,
            "agent_outputs": transparency_report,
            "recommendations": strategy_output.changes.get("recommendations", []),
            "warnings": strategy_output.changes.get("warnings", []),
            "tradeoffs": strategy_output.changes.get("tradeoffs", []),
            "strategic_priority": strategy_output.changes.get("strategic_priority", "OPTIMIZATION"),
            "total_rules_triggered": sum(len(o.rules_triggered) for o in agent_outputs.values()),
            "confidence": round(sum(o.confidence for o in agent_outputs.values()) / len(agent_outputs), 1)
        }
    
    def _calculate_health(self, state: Dict[str, float]) -> float:
        """Calculate Business Health Index (0-100)"""
        components = []
        
        # Revenue factor (25%)
        revenue = state.get("revenue", 100000)
        revenue_score = min(100, (revenue / 100000) * 50 + 50)
        components.append(revenue_score * 0.25)
        
        # Sentiment factor (25%)
        sentiment = state.get("sentiment", 75)
        components.append(sentiment * 0.25)
        
        # Risk factor (25%) - inverted
        risk = state.get("risk_score", 50)
        risk_score = 100 - risk
        components.append(risk_score * 0.25)
        
        # Customer factor (25%)
        churn = state.get("churn_rate", 5)
        customer_health = max(0, 100 - churn * 5)
        components.append(customer_health * 0.25)
        
        return round(sum(components), 1)


# Global engine instance
multi_agent_engine = MultiAgentEngine()
