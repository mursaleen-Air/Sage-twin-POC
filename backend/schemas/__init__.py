"""
Schemas Package
Data validation schemas for department data.
"""

from .department_schemas import (
    DepartmentType,
    TimeSeriesDataPoint,
    MetricDefinition,
    RevenueData,
    CustomerData,
    OperationsData,
    MarketingData,
    HRData,
    SupplyChainData,
    CompanySnapshot,
    METRIC_CATALOG
)

__all__ = [
    "DepartmentType",
    "TimeSeriesDataPoint",
    "MetricDefinition",
    "RevenueData",
    "CustomerData",
    "OperationsData",
    "MarketingData",
    "HRData",
    "SupplyChainData",
    "CompanySnapshot",
    "METRIC_CATALOG"
]
