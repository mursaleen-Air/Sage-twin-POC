# semantic_agents.py
"""
Semantic AI Agents - OpenAI-powered business advisors
Each agent specializes in different aspects of business analysis
"""

import os
from openai import OpenAI
from typing import Dict, List, Any
from knowledge_graph import (
    BUSINESS_KNOWLEDGE_GRAPH, 
    STRATEGIC_ACTIONS, 
    get_cascading_effects,
    get_recommended_actions
)
from memory import conversation_memory

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def parse_user_intent(message: str, company_data: Dict) -> Dict[str, Any]:
    """
    Use OpenAI to understand user intent and extract goals
    """
    metrics_context = ", ".join([f"{k}: {v}" for k, v in company_data.items() 
                                  if isinstance(v, (int, float))])
    
    prompt = f"""Analyze this business query and extract the user's intent.

User Message: "{message}"

Available Company Metrics: {metrics_context}

Extract:
1. primary_goal: The main business objective (e.g., "increase_customers", "reduce_churn", "increase_revenue")
2. target_metric: The specific metric they want to change (if mentioned)
3. target_change: The desired change amount/percentage (if mentioned)
4. constraints: Any constraints or limitations mentioned
5. timeframe: Any timeline mentioned

Respond in JSON format only:
{{"primary_goal": "...", "target_metric": "...", "target_change": null, "constraints": [], "timeframe": null}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a business intent parser. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "primary_goal": "general_advice",
            "target_metric": None,
            "target_change": None,
            "constraints": [],
            "timeframe": None
        }


def strategy_agent(intent: Dict, company_data: Dict, memory_context: str) -> str:
    """
    Strategic advisor agent - provides high-level recommendations
    """
    goal = intent.get("primary_goal", "general")
    target = intent.get("target_metric")
    
    # Get cascading effects if target specified
    cascading = {}
    if target and target in company_data:
        change = intent.get("target_change", 10)
        cascading = get_cascading_effects(target, float(change) if change else 10)
    
    # Get recommended actions
    actions = get_recommended_actions(goal)
    actions_text = "\n".join([
        f"- {a['description']}: Timeframe {a['timeframe']}, "
        f"Primary effects: {a['primary_effects']}" 
        for a in actions[:3]
    ])
    
    cascading_text = "\n".join([f"- {k}: {v:+.1f}%" for k, v in cascading.items()]) if cascading else "N/A"
    
    prompt = f"""You are a strategic business advisor for SAGE-Twin digital twin system.

CONTEXT:
{memory_context}

USER GOAL: {goal}
TARGET METRIC: {target or 'Not specified'}

CASCADING EFFECTS (from knowledge graph):
{cascading_text}

RECOMMENDED ACTIONS:
{actions_text}

Provide strategic advice that:
1. Acknowledges the user's goal
2. Explains how achieving this goal will affect other metrics (use the cascading effects)
3. Recommends 2-3 specific steps with clear rationale
4. Mentions potential risks or trade-offs
5. Suggests a realistic timeframe

Be conversational but professional. Keep response under 200 words."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful strategic business advisor. Be specific and actionable."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Strategic analysis unavailable: {str(e)}"


def impact_agent(target_metric: str, change: float, company_data: Dict) -> Dict[str, Any]:
    """
    Impact analysis agent - calculates cascading effects
    """
    # Get cascading effects from knowledge graph
    effects = get_cascading_effects(target_metric, change)
    
    # Calculate projected values
    projections = {}
    for metric, effect in effects.items():
        if metric in company_data and isinstance(company_data[metric], (int, float)):
            original = company_data[metric]
            projected = original * (1 + effect / 100)
            projections[metric] = {
                "original": original,
                "projected": round(projected, 2),
                "change_percent": round(effect, 2)
            }
    
    # Determine overall risk
    negative_effects = [e for e in effects.values() if e < 0]
    risk_score = min(100, abs(sum(negative_effects)) * 2) if negative_effects else 0
    
    return {
        "effects": projections,
        "risk_score": round(risk_score, 1),
        "affected_metrics": len(projections),
        "primary_change": {target_metric: change}
    }


def action_agent(goal: str, company_data: Dict, memory_context: str) -> List[Dict[str, Any]]:
    """
    Action planning agent - provides specific actionable steps
    """
    actions = get_recommended_actions(goal)
    
    # Use AI to prioritize and contextualize actions
    actions_json = str([{k: v for k, v in a.items() if k != 'risks'} for a in actions])
    
    prompt = f"""Based on the goal "{goal}" and these potential actions:
{actions_json}

And this company context:
{memory_context}

Rank the top 3 actions and for each provide:
1. Why it's suitable for this specific situation
2. First concrete step to take
3. Expected outcome

Respond as a JSON array:
[{{"action": "name", "reasoning": "...", "first_step": "...", "expected_outcome": "..."}}]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a business action planner. Respond only with valid JSON array."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.5
        )
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        # Fallback to basic actions
        return [{"action": a["description"], "reasoning": "Recommended action", 
                 "first_step": "Analyze current state", "expected_outcome": str(a["primary_effects"])} 
                for a in actions[:3]]


def chat_with_agents(message: str, company_data: Dict) -> Dict[str, Any]:
    """
    Main chat interface - orchestrates all agents
    """
    # Get memory context
    memory_context = conversation_memory.get_context_summary()
    chat_history = conversation_memory.get_chat_history(limit=5)
    
    # Store user message
    conversation_memory.add_message("user", message)
    
    # Parse user intent
    intent = parse_user_intent(message, company_data)
    
    # Store goal if identified
    if intent.get("primary_goal"):
        conversation_memory.add_goal(intent["primary_goal"])
    
    # Get strategic advice
    strategy_response = strategy_agent(intent, company_data, memory_context)
    
    # Get impact analysis if target specified
    impact = None
    if intent.get("target_metric") and intent["target_metric"] in company_data:
        change = intent.get("target_change", 10)
        impact = impact_agent(intent["target_metric"], float(change) if change else 10, company_data)
    
    # Get action plan
    action_plan = action_agent(intent.get("primary_goal", "improve_business"), company_data, memory_context)
    
    # Store assistant response
    conversation_memory.add_message("assistant", strategy_response, {"intent": intent})
    
    # Extract and store insights
    if impact and impact.get("affected_metrics", 0) > 2:
        insight = f"Changing {intent.get('target_metric')} affects {impact['affected_metrics']} other metrics"
        conversation_memory.add_insight(insight)
    
    return {
        "intent": intent,
        "response": strategy_response,
        "impact_analysis": impact,
        "action_plan": action_plan,
        "context": {
            "goals_tracked": conversation_memory.goals,
            "insights": conversation_memory.insights[-3:],
            "message_count": len(conversation_memory.messages)
        }
    }
