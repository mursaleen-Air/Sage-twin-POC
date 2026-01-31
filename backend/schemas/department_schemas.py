"""
Department Data Schemas
Defines structured schemas for each business department with validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class DepartmentType(str, Enum):
    REVENUE = "revenue"
    OPERATIONS = "operations"
    MARKETING = "marketing"
    CUSTOMERS = "customers"
    HR = "hr"
    SUPPLY_CHAIN = "supply_chain"


# ============================================
# BASE SCHEMAS
# ============================================

class TimeSeriesDataPoint(BaseModel):
    """Single data point in a time series"""
    timestamp: datetime
    value: float
    metadata: Optional[Dict[str, Any]] = None


class MetricDefinition(BaseModel):
    """Definition of a business metric"""
    name: str
    description: str
    unit: str  # e.g., "USD", "count", "percentage", "days"
    department: DepartmentType
    is_kpi: bool = False
    target_direction: str = "up"  # "up", "down", "stable"
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None


# ============================================
# DEPARTMENT-SPECIFIC SCHEMAS
# ============================================

class RevenueData(BaseModel):
    """Revenue & Financial Department Data"""
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Core Metrics
    revenue: float = Field(..., ge=0, description="Total revenue")
    profit: Optional[float] = Field(None, description="Net profit")
    gross_margin: Optional[float] = Field(None, ge=0, le=100, description="Gross margin %")
    operating_costs: Optional[float] = Field(None, ge=0, description="Operating costs")
    
    # Growth Metrics
    revenue_growth_pct: Optional[float] = Field(None, description="Revenue growth %")
    yoy_growth: Optional[float] = Field(None, description="Year-over-year growth %")
    
    # Pricing
    average_order_value: Optional[float] = Field(None, ge=0, description="AOV")
    price_per_unit: Optional[float] = Field(None, ge=0)
    
    # Cash Flow
    cash_flow: Optional[float] = Field(None)
    accounts_receivable: Optional[float] = Field(None, ge=0)
    accounts_payable: Optional[float] = Field(None, ge=0)


class CustomerData(BaseModel):
    """Customer Department Data"""
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Core Metrics
    total_customers: int = Field(..., ge=0)
    new_customers: Optional[int] = Field(None, ge=0)
    churned_customers: Optional[int] = Field(None, ge=0)
    
    # Rates
    churn_rate: Optional[float] = Field(None, ge=0, le=100)
    retention_rate: Optional[float] = Field(None, ge=0, le=100)
    conversion_rate: Optional[float] = Field(None, ge=0, le=100)
    
    # Value Metrics
    customer_lifetime_value: Optional[float] = Field(None, ge=0)
    average_revenue_per_user: Optional[float] = Field(None, ge=0)
    
    # Satisfaction
    nps_score: Optional[float] = Field(None, ge=-100, le=100)
    csat_score: Optional[float] = Field(None, ge=0, le=100)
    sentiment_score: Optional[float] = Field(None, ge=0, le=100)


class OperationsData(BaseModel):
    """Operations & Logistics Department Data"""
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Delivery
    average_delivery_time: Optional[float] = Field(None, ge=0, description="Days")
    on_time_delivery_rate: Optional[float] = Field(None, ge=0, le=100)
    delivery_delay_days: Optional[float] = Field(None, ge=0)
    
    # Fulfillment
    fulfillment_rate: Optional[float] = Field(None, ge=0, le=100)
    order_accuracy_rate: Optional[float] = Field(None, ge=0, le=100)
    
    # Inventory
    inventory_level: Optional[int] = Field(None, ge=0)
    inventory_turnover: Optional[float] = Field(None, ge=0)
    stockout_rate: Optional[float] = Field(None, ge=0, le=100)
    
    # Returns
    return_rate: Optional[float] = Field(None, ge=0, le=100)
    defect_rate: Optional[float] = Field(None, ge=0, le=100)
    
    # Costs
    shipping_cost: Optional[float] = Field(None, ge=0)
    operational_cost: Optional[float] = Field(None, ge=0)


class MarketingData(BaseModel):
    """Marketing Department Data"""
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Spend
    marketing_spend: float = Field(..., ge=0)
    ad_spend: Optional[float] = Field(None, ge=0)
    
    # Performance
    impressions: Optional[int] = Field(None, ge=0)
    clicks: Optional[int] = Field(None, ge=0)
    conversions: Optional[int] = Field(None, ge=0)
    
    # Rates
    click_through_rate: Optional[float] = Field(None, ge=0, le=100)
    conversion_rate: Optional[float] = Field(None, ge=0, le=100)
    
    # Costs
    cost_per_click: Optional[float] = Field(None, ge=0)
    cost_per_acquisition: Optional[float] = Field(None, ge=0)
    customer_acquisition_cost: Optional[float] = Field(None, ge=0)
    
    # ROI
    marketing_roi: Optional[float] = Field(None, description="Return on marketing investment %")
    roas: Optional[float] = Field(None, ge=0, description="Return on ad spend")


class HRData(BaseModel):
    """Human Resources Department Data"""
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Headcount
    total_employees: int = Field(..., ge=0)
    new_hires: Optional[int] = Field(None, ge=0)
    terminations: Optional[int] = Field(None, ge=0)
    
    # Rates
    turnover_rate: Optional[float] = Field(None, ge=0, le=100)
    retention_rate: Optional[float] = Field(None, ge=0, le=100)
    
    # Costs
    payroll_cost: Optional[float] = Field(None, ge=0)
    cost_per_hire: Optional[float] = Field(None, ge=0)
    training_cost: Optional[float] = Field(None, ge=0)
    
    # Productivity
    revenue_per_employee: Optional[float] = Field(None, ge=0)
    employee_satisfaction: Optional[float] = Field(None, ge=0, le=100)
    
    # Time
    average_tenure_months: Optional[float] = Field(None, ge=0)
    time_to_fill_days: Optional[float] = Field(None, ge=0)


class SupplyChainData(BaseModel):
    """Supply Chain Department Data"""
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Suppliers
    supplier_count: Optional[int] = Field(None, ge=0)
    supplier_lead_time_days: Optional[float] = Field(None, ge=0)
    supplier_on_time_rate: Optional[float] = Field(None, ge=0, le=100)
    
    # Procurement
    procurement_cost: Optional[float] = Field(None, ge=0)
    cost_savings: Optional[float] = Field(None)
    
    # Inventory
    raw_material_inventory: Optional[float] = Field(None, ge=0)
    work_in_progress: Optional[float] = Field(None, ge=0)
    finished_goods_inventory: Optional[float] = Field(None, ge=0)
    
    # Performance
    order_cycle_time_days: Optional[float] = Field(None, ge=0)
    perfect_order_rate: Optional[float] = Field(None, ge=0, le=100)


# ============================================
# AGGREGATED COMPANY DATA
# ============================================

class CompanySnapshot(BaseModel):
    """Complete company state at a point in time"""
    timestamp: datetime = Field(default_factory=datetime.now)
    
    revenue: Optional[RevenueData] = None
    customers: Optional[CustomerData] = None
    operations: Optional[OperationsData] = None
    marketing: Optional[MarketingData] = None
    hr: Optional[HRData] = None
    supply_chain: Optional[SupplyChainData] = None
    
    # Computed
    health_score: Optional[float] = Field(None, ge=0, le=100)
    risk_score: Optional[float] = Field(None, ge=0, le=100)
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert nested structure to flat dictionary for ML processing"""
        flat = {"timestamp": self.timestamp}
        
        for dept_name in ["revenue", "customers", "operations", "marketing", "hr", "supply_chain"]:
            dept_data = getattr(self, dept_name)
            if dept_data:
                for key, value in dept_data.dict().items():
                    if key != "timestamp" and value is not None:
                        flat[f"{dept_name}_{key}"] = value
        
        if self.health_score is not None:
            flat["health_score"] = self.health_score
        if self.risk_score is not None:
            flat["risk_score"] = self.risk_score
            
        return flat


# ============================================
# METRIC CATALOG
# ============================================

METRIC_CATALOG: Dict[str, MetricDefinition] = {
    "revenue": MetricDefinition(
        name="Revenue",
        description="Total revenue generated",
        unit="USD",
        department=DepartmentType.REVENUE,
        is_kpi=True,
        target_direction="up"
    ),
    "profit": MetricDefinition(
        name="Profit",
        description="Net profit after expenses",
        unit="USD",
        department=DepartmentType.REVENUE,
        is_kpi=True,
        target_direction="up"
    ),
    "customers": MetricDefinition(
        name="Total Customers",
        description="Active customer count",
        unit="count",
        department=DepartmentType.CUSTOMERS,
        is_kpi=True,
        target_direction="up"
    ),
    "churn_rate": MetricDefinition(
        name="Churn Rate",
        description="Customer churn percentage",
        unit="percentage",
        department=DepartmentType.CUSTOMERS,
        is_kpi=True,
        target_direction="down",
        warning_threshold=5.0,
        critical_threshold=10.0
    ),
    "sentiment": MetricDefinition(
        name="Customer Sentiment",
        description="Overall customer sentiment score",
        unit="score",
        department=DepartmentType.CUSTOMERS,
        is_kpi=True,
        target_direction="up",
        warning_threshold=60.0,
        critical_threshold=40.0
    ),
    "delivery_delay": MetricDefinition(
        name="Delivery Delay",
        description="Average delivery delay in days",
        unit="days",
        department=DepartmentType.OPERATIONS,
        is_kpi=True,
        target_direction="down",
        warning_threshold=3.0,
        critical_threshold=5.0
    ),
    "marketing_spend": MetricDefinition(
        name="Marketing Spend",
        description="Total marketing expenditure",
        unit="USD",
        department=DepartmentType.MARKETING,
        target_direction="stable"
    ),
    "cac": MetricDefinition(
        name="Customer Acquisition Cost",
        description="Cost to acquire one customer",
        unit="USD",
        department=DepartmentType.MARKETING,
        is_kpi=True,
        target_direction="down"
    ),
}
