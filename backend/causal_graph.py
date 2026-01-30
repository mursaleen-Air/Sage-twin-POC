# causal_graph.py
"""
Step 3: Causal Relationship Map
The internal business graph that makes this a Digital Twin, not just a dashboard.
"""

from typing import Dict, List, Tuple, Any


# ============================================
# CAUSAL RELATIONSHIP MAP
# ============================================
# Format: source -> [(target, effect_type, multiplier)]
# multiplier: how much 1% change in source affects target
# positive multiplier = same direction, negative = inverse

CAUSAL_MAP = {
    # Marketing effects
    "marketing_spend": [
        ("customers", "increases", 0.4),      # Marketing ↑ → Customers ↑
        ("brand_awareness", "increases", 0.6),
        ("revenue", "short_term_cost", -0.1), # Costs money initially
    ],
    
    # Customer effects
    "customers": [
        ("revenue", "generates", 0.7),        # Customers ↑ → Revenue ↑
        ("support_load", "increases", 0.5),
        ("word_of_mouth", "amplifies", 0.3),
    ],
    
    # Revenue effects
    "revenue": [
        ("profit", "increases", 0.6),
        ("growth_capacity", "enables", 0.4),
        ("risk_score", "reduces", -0.2),      # More revenue = less risk
    ],
    
    # Delivery delay effects (CRITICAL PATH)
    "delivery_delay": [
        ("sentiment", "hurts", -0.5),         # Delay ↑ → Sentiment ↓
        ("customer_satisfaction", "hurts", -0.6),
        ("churn_rate", "increases", 0.4),     # Delay ↑ → Churn ↑
        ("operational_cost", "may_reduce", -0.1),
    ],
    
    # Sentiment effects
    "sentiment": [
        ("churn_rate", "inversely_affects", -0.4),  # Sentiment ↓ → Churn ↑
        ("customers", "attracts", 0.3),
        ("brand_reputation", "reflects", 0.5),
        ("revenue", "via_retention", 0.3),    # Sentiment → Revenue via loyalty
    ],
    
    # Churn effects
    "churn_rate": [
        ("customers", "reduces", -0.6),       # Churn ↑ → Customers ↓
        ("revenue", "reduces", -0.5),         # Churn ↑ → Revenue ↓
        ("acquisition_cost", "increases_need", 0.3),
    ],
    
    # Cost effects
    "costs": [
        ("profit", "reduces", -0.8),          # Costs ↑ → Profit ↓
        ("risk_score", "increases", 0.3),     # Costs ↑ → Risk ↑
    ],
    
    # Price effects
    "price": [
        ("revenue", "can_increase", 0.4),     # Price ↑ → Revenue might ↑
        ("customers", "may_reduce", -0.2),    # Price ↑ → Some customers leave
        ("sentiment", "may_decrease", -0.15),
        ("churn_rate", "may_increase", 0.15),
    ],
    
    # Profit effects
    "profit": [
        ("risk_score", "reduces", -0.4),      # Profit ↑ → Risk ↓
        ("growth_capacity", "enables", 0.5),
        ("investment_capacity", "enables", 0.4),
    ],
    
    # Satisfaction effects
    "customer_satisfaction": [
        ("sentiment", "correlates", 0.8),
        ("churn_rate", "reduces", -0.5),
        ("word_of_mouth", "increases", 0.4),
        ("customers", "attracts", 0.25),
    ],
    
    # Risk effects  
    "risk_score": [
        ("investor_confidence", "reduces", -0.5),
        ("growth_capacity", "limits", -0.3),
    ],
}


# ============================================
# PROPAGATION ENGINE
# ============================================

def propagate_change(
    source_metric: str, 
    change_percent: float, 
    current_state: Dict[str, float],
    max_depth: int = 4
) -> Dict[str, Any]:
    """
    Step 6: State Propagation Logic
    When one metric changes, it ripples through the system.
    Returns all affected metrics with their calculated changes.
    """
    
    # Track all effects
    effects: Dict[str, float] = {source_metric: change_percent}
    propagation_chain: List[Dict[str, Any]] = []
    visited = {source_metric}
    
    # BFS propagation
    queue: List[Tuple[str, float, int, str]] = [(source_metric, change_percent, 0, "initial")]
    
    while queue:
        current, current_change, depth, trigger = queue.pop(0)
        
        if depth >= max_depth:
            continue
        
        # Get downstream effects
        if current in CAUSAL_MAP:
            for target, effect_type, multiplier in CAUSAL_MAP[current]:
                # Calculate propagated effect
                effect = current_change * multiplier
                
                # Apply diminishing returns with depth
                effect *= (0.75 ** depth)
                
                # Only propagate meaningful changes
                if abs(effect) >= 0.1:
                    # Accumulate effects
                    effects[target] = effects.get(target, 0) + effect
                    
                    # Record chain
                    propagation_chain.append({
                        "from": current,
                        "to": target,
                        "effect_type": effect_type,
                        "change": round(effect, 2),
                        "depth": depth + 1,
                        "rule": f"{current} → {target} ({effect_type})"
                    })
                    
                    # Continue propagation if not visited
                    if target not in visited:
                        visited.add(target)
                        queue.append((target, effect, depth + 1, current))
    
    # Calculate new state values
    new_state: Dict[str, float] = current_state.copy()
    for metric, change in effects.items():
        if metric in current_state:
            original = current_state[metric]
            new_state[metric] = round(original * (1 + change / 100), 2)
    
    return {
        "effects": effects,
        "new_state": new_state,
        "propagation_chain": propagation_chain,
        "metrics_affected": len(effects),
        "propagation_depth": max(p["depth"] for p in propagation_chain) if propagation_chain else 0
    }


def get_causal_graph_visualization() -> Dict[str, Any]:
    """Get graph structure for frontend visualization"""
    nodes = set()
    edges = []
    
    for source, relationships in CAUSAL_MAP.items():
        nodes.add(source)
        for target, effect_type, multiplier in relationships:
            nodes.add(target)
            edges.append({
                "source": source,
                "target": target,
                "type": effect_type,
                "strength": abs(multiplier),
                "direction": "positive" if multiplier > 0 else "negative"
            })
    
    return {
        "nodes": [{"id": n, "label": n.replace("_", " ").title()} for n in nodes],
        "edges": edges,
        "total_relationships": len(edges)
    }


def explain_relationship(source: str, target: str) -> str:
    """Explain the causal relationship between two metrics"""
    if source in CAUSAL_MAP:
        for t, effect_type, multiplier in CAUSAL_MAP[source]:
            if t == target:
                direction = "increases" if multiplier > 0 else "decreases"
                strength = "strongly" if abs(multiplier) > 0.5 else "moderately" if abs(multiplier) > 0.25 else "slightly"
                return f"When {source.replace('_', ' ')} changes, it {strength} {direction} {target.replace('_', ' ')} ({effect_type} relationship)"
    
    return f"No direct causal relationship found between {source} and {target}"
