# knowledge_graph.py
"""
Business Knowledge Graph - Semantic relationships between metrics
Used to derive implicit knowledge and cascading effects
"""

# Define relationships between business metrics
# Format: source -> [(target, relationship_type, strength, direction)]
# strength: how much 1% change in source affects target (multiplier)
# direction: "positive" (same direction) or "negative" (inverse)

BUSINESS_KNOWLEDGE_GRAPH = {
    "revenue": [
        ("profit", "increases", 0.8, "positive"),
        ("marketing_budget", "enables", 0.3, "positive"),
        ("customer_satisfaction", "funds_improvement", 0.2, "positive"),
        ("growth_capacity", "enables", 0.5, "positive"),
    ],
    "customers": [
        ("revenue", "generates", 0.7, "positive"),
        ("customer_support_load", "increases", 0.8, "positive"),
        ("word_of_mouth", "amplifies", 0.4, "positive"),
        ("churn_rate", "can_increase", 0.1, "positive"),
    ],
    "marketing_budget": [
        ("customers", "acquires", 0.4, "positive"),
        ("brand_awareness", "increases", 0.6, "positive"),
        ("revenue", "short_term_reduces", 0.1, "negative"),
        ("customer_acquisition_cost", "increases", 0.2, "positive"),
    ],
    "customer_satisfaction": [
        ("churn_rate", "reduces", 0.5, "negative"),
        ("word_of_mouth", "increases", 0.6, "positive"),
        ("customers", "attracts", 0.3, "positive"),
        ("revenue", "increases_via_retention", 0.4, "positive"),
    ],
    "sentiment": [
        ("customer_satisfaction", "correlates", 0.8, "positive"),
        ("churn_rate", "inversely_affects", 0.4, "negative"),
        ("brand_reputation", "reflects", 0.7, "positive"),
    ],
    "delivery_delay": [
        ("customer_satisfaction", "hurts", 0.6, "negative"),
        ("sentiment", "decreases", 0.5, "negative"),
        ("churn_rate", "increases", 0.4, "positive"),
        ("operational_cost", "may_reduce", 0.2, "negative"),
    ],
    "churn_rate": [
        ("customers", "reduces", 0.8, "negative"),
        ("revenue", "decreases", 0.6, "negative"),
        ("customer_acquisition_cost", "increases_need", 0.3, "positive"),
    ],
    "price": [
        ("revenue", "can_increase", 0.5, "positive"),
        ("customer_satisfaction", "may_decrease", 0.3, "negative"),
        ("churn_rate", "may_increase", 0.2, "positive"),
        ("competitive_position", "may_weaken", 0.2, "negative"),
    ],
    "employee_satisfaction": [
        ("productivity", "increases", 0.5, "positive"),
        ("customer_satisfaction", "improves", 0.3, "positive"),
        ("turnover_rate", "reduces", 0.6, "negative"),
    ],
    "operational_efficiency": [
        ("delivery_delay", "reduces", 0.5, "negative"),
        ("cost", "reduces", 0.4, "negative"),
        ("customer_satisfaction", "improves", 0.3, "positive"),
    ],
}

# Strategic actions and their effects
STRATEGIC_ACTIONS = {
    "increase_marketing": {
        "description": "Increase marketing spend and campaigns",
        "primary_effects": {"marketing_budget": 20, "customers": 10, "brand_awareness": 15},
        "secondary_effects": {"revenue": 8, "customer_acquisition_cost": 5},
        "risks": ["ROI may vary", "Market saturation possible"],
        "timeframe": "3-6 months",
    },
    "improve_customer_service": {
        "description": "Enhance customer support and experience",
        "primary_effects": {"customer_satisfaction": 15, "sentiment": 12},
        "secondary_effects": {"churn_rate": -8, "word_of_mouth": 10, "customers": 5},
        "risks": ["Increased operational costs", "Training time needed"],
        "timeframe": "2-4 months",
    },
    "optimize_pricing": {
        "description": "Strategic price optimization",
        "primary_effects": {"revenue": 10, "profit_margin": 8},
        "secondary_effects": {"customer_satisfaction": -3, "competitive_position": -2},
        "risks": ["Customer backlash", "Competitor response"],
        "timeframe": "1-2 months",
    },
    "reduce_delivery_time": {
        "description": "Streamline logistics and delivery",
        "primary_effects": {"delivery_delay": -30, "customer_satisfaction": 12},
        "secondary_effects": {"sentiment": 8, "churn_rate": -5, "operational_cost": 5},
        "risks": ["Higher logistics costs", "Quality control challenges"],
        "timeframe": "2-6 months",
    },
    "launch_referral_program": {
        "description": "Implement customer referral incentives",
        "primary_effects": {"customers": 15, "customer_acquisition_cost": -20},
        "secondary_effects": {"revenue": 10, "word_of_mouth": 25},
        "risks": ["Program abuse", "Margin impact from incentives"],
        "timeframe": "1-3 months",
    },
    "product_improvement": {
        "description": "Enhance product quality and features",
        "primary_effects": {"customer_satisfaction": 20, "competitive_position": 15},
        "secondary_effects": {"churn_rate": -10, "revenue": 12, "price_elasticity": 10},
        "risks": ["Development costs", "Feature creep"],
        "timeframe": "3-9 months",
    },
    "expand_market": {
        "description": "Enter new markets or segments",
        "primary_effects": {"customers": 25, "revenue": 20},
        "secondary_effects": {"brand_awareness": 15, "operational_complexity": 20},
        "risks": ["Market fit uncertainty", "Resource strain"],
        "timeframe": "6-12 months",
    },
    "cost_optimization": {
        "description": "Reduce operational costs",
        "primary_effects": {"operational_cost": -15, "profit_margin": 10},
        "secondary_effects": {"delivery_delay": 5, "employee_satisfaction": -5},
        "risks": ["Quality degradation", "Employee morale"],
        "timeframe": "2-4 months",
    },
}

# Goal to action mapping
GOAL_ACTION_MAP = {
    "increase_customers": ["increase_marketing", "launch_referral_program", "expand_market", "improve_customer_service"],
    "increase_revenue": ["optimize_pricing", "increase_marketing", "expand_market", "product_improvement"],
    "reduce_churn": ["improve_customer_service", "product_improvement", "reduce_delivery_time"],
    "improve_satisfaction": ["improve_customer_service", "reduce_delivery_time", "product_improvement"],
    "increase_profit": ["optimize_pricing", "cost_optimization", "reduce_churn"],
    "grow_market_share": ["expand_market", "increase_marketing", "optimize_pricing"],
    "improve_efficiency": ["cost_optimization", "reduce_delivery_time"],
}


def get_cascading_effects(metric: str, change_percent: float, depth: int = 3) -> dict:
    """
    Calculate cascading effects through the knowledge graph
    Returns a dict of all affected metrics with their projected changes
    """
    effects = {metric: change_percent}
    visited = {metric}
    queue = [(metric, change_percent, 0)]
    
    while queue:
        current, current_change, current_depth = queue.pop(0)
        
        if current_depth >= depth:
            continue
            
        if current in BUSINESS_KNOWLEDGE_GRAPH:
            for target, relationship, strength, direction in BUSINESS_KNOWLEDGE_GRAPH[current]:
                if target not in visited:
                    # Calculate effect based on strength and direction
                    effect = current_change * strength
                    if direction == "negative":
                        effect = -effect
                    
                    # Diminish effect with depth
                    effect *= (0.7 ** current_depth)
                    
                    if abs(effect) > 0.5:  # Only track meaningful effects
                        effects[target] = effects.get(target, 0) + effect
                        visited.add(target)
                        queue.append((target, effect, current_depth + 1))
    
    return effects


def get_recommended_actions(goal: str) -> list:
    """Get recommended strategic actions for a given goal"""
    # Normalize goal
    goal_lower = goal.lower().replace(" ", "_")
    
    # Find matching goal
    for key in GOAL_ACTION_MAP:
        if key in goal_lower or goal_lower in key:
            action_ids = GOAL_ACTION_MAP[key]
            return [STRATEGIC_ACTIONS[aid] for aid in action_ids if aid in STRATEGIC_ACTIONS]
    
    # Default recommendations
    return list(STRATEGIC_ACTIONS.values())[:3]
