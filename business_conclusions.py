"""
Business Conclusions Module
Provides recommendations engine, financial analysis, and alerts system
for demand predictions
"""

import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple


class DemandLevel(Enum):
    """Demand classification levels"""
    VERY_LOW = "Very Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "🔴 CRITICAL"
    WARNING = "🟡 WARNING"
    INFO = "ℹ️ INFO"
    SUCCESS = "✅ SUCCESS"


@dataclass
class Alert:
    """Alert data structure"""
    severity: AlertSeverity
    title: str
    message: str
    recommendation: str


@dataclass
class Recommendation:
    """Recommendation data structure"""
    action: str
    priority: str  # HIGH, MEDIUM, LOW
    impact: str
    details: str


class BusinessAnalyzer:
    """Analyzes demand predictions and provides business insights"""
    
    def __init__(self, unit_price: float, country: str, stock_code: str, 
                 predicted_quantity: float, avg_historical_quantity: float,
                 avg_unit_price: float = None, inventory_cost_per_unit: float = 0.5):
        """
        Initialize Business Analyzer
        
        Args:
            unit_price: Current unit price
            country: Country of sale
            stock_code: Product stock code
            predicted_quantity: Predicted demand quantity
            avg_historical_quantity: Average historical quantity sold
            avg_unit_price: Average unit price (for comparisons)
            inventory_cost_per_unit: Cost to hold one unit per period
        """
        self.unit_price = unit_price
        self.country = country
        self.stock_code = stock_code
        self.predicted_quantity = max(0, predicted_quantity)  # Ensure non-negative
        self.avg_historical_quantity = avg_historical_quantity
        self.avg_unit_price = avg_unit_price or unit_price
        self.inventory_cost_per_unit = inventory_cost_per_unit
        
    def classify_demand(self) -> Tuple[DemandLevel, float]:
        """
        Classify demand based on historical average and prediction
        
        Returns:
            Tuple of (DemandLevel, confidence_score 0-1)
        """
        if self.avg_historical_quantity == 0:
            ratio = 1.0
        else:
            ratio = self.predicted_quantity / self.avg_historical_quantity
        
        if ratio >= 2.0:
            return DemandLevel.VERY_HIGH, min(0.95, ratio / 3.0)
        elif ratio >= 1.5:
            return DemandLevel.HIGH, 0.85
        elif ratio >= 0.75:
            return DemandLevel.MEDIUM, 0.80
        elif ratio >= 0.25:
            return DemandLevel.LOW, 0.70
        else:
            return DemandLevel.VERY_LOW, 0.60
    
    def calculate_financial_impact(self) -> Dict:
        """Calculate financial metrics for the prediction"""
        predicted_revenue = self.predicted_quantity * self.unit_price
        historical_revenue = self.avg_historical_quantity * self.avg_unit_price
        revenue_variance = predicted_revenue - historical_revenue
        revenue_variance_pct = (revenue_variance / historical_revenue * 100) if historical_revenue > 0 else 0
        
        # Inventory holding cost
        holding_cost = self.predicted_quantity * self.inventory_cost_per_unit
        
        # Margin assumption (typically 30-40%)
        estimated_margin = predicted_revenue * 0.35
        profit_potential = estimated_margin - holding_cost
        
        return {
            'predicted_revenue': predicted_revenue,
            'historical_revenue': historical_revenue,
            'revenue_variance': revenue_variance,
            'revenue_variance_pct': revenue_variance_pct,
            'holding_cost': holding_cost,
            'estimated_profit': profit_potential,
            'profit_margin_pct': (profit_potential / predicted_revenue * 100) if predicted_revenue > 0 else 0,
            'roi': (profit_potential / holding_cost * 100) if holding_cost > 0 else 0
        }
    
    def generate_recommendations(self) -> List[Recommendation]:
        """Generate actionable recommendations based on prediction"""
        recommendations = []
        demand_level, confidence = self.classify_demand()
        financials = self.calculate_financial_impact()
        
        # Demand-based recommendations
        if demand_level == DemandLevel.VERY_HIGH:
            recommendations.append(Recommendation(
                action="URGENT RESTOCK",
                priority="HIGH",
                impact="Prevent stockouts and maximize revenue",
                details=f"Predicted demand is {self.predicted_quantity/self.avg_historical_quantity:.1f}x historical average. "
                       f"Recommended to increase inventory immediately. Expected revenue: £{financials['predicted_revenue']:.2f}"
            ))
            recommendations.append(Recommendation(
                action="SCALE PRODUCTION",
                priority="HIGH",
                impact="Meet surge in customer demand",
                details=f"Coordinate with suppliers to increase order quantities by {((self.predicted_quantity/self.avg_historical_quantity - 1) * 100):.0f}%"
            ))
            recommendations.append(Recommendation(
                action="PREMIUM PRICING OPPORTUNITY",
                priority="MEDIUM",
                impact="Capitalize on high demand with potential price increase",
                details=f"Consider modest price increase (5-10%) during high-demand period. Current unit price: £{self.unit_price:.2f}"
            ))
        
        elif demand_level == DemandLevel.HIGH:
            recommendations.append(Recommendation(
                action="INCREASE STOCK LEVELS",
                priority="MEDIUM",
                impact="Ensure sufficient inventory to meet expected demand",
                details=f"Increase inventory by approximately {((self.predicted_quantity/self.avg_historical_quantity - 1) * 100):.0f}%"
            ))
            recommendations.append(Recommendation(
                action="MONITOR INVENTORY CLOSELY",
                priority="MEDIUM",
                impact="Prevent unexpected stockouts",
                details="Set up inventory alerts for this product"
            ))
        
        elif demand_level == DemandLevel.MEDIUM:
            recommendations.append(Recommendation(
                action="MAINTAIN CURRENT STOCK",
                priority="LOW",
                impact="Stable inventory management",
                details="Current inventory levels appear appropriate for expected demand"
            ))
        
        elif demand_level == DemandLevel.LOW:
            recommendations.append(Recommendation(
                action="PROMOTIONAL CAMPAIGN",
                priority="MEDIUM",
                impact="Boost sales of slow-moving product",
                details=f"Predicted demand is {(1 - self.predicted_quantity/self.avg_historical_quantity) * 100:.0f}% below average. "
                       f"Consider discounts or bundle deals."
            ))
            recommendations.append(Recommendation(
                action="REDUCE ORDER QUANTITY",
                priority="MEDIUM",
                impact="Minimize inventory holding costs",
                details=f"Reduce incoming orders to prevent excess inventory and associated costs"
            ))
        
        elif demand_level == DemandLevel.VERY_LOW:
            recommendations.append(Recommendation(
                action="CLEARANCE SALE / DISCONTINUATION",
                priority="HIGH",
                impact="Minimize losses on dead inventory",
                details=f"Predicted demand critically low ({self.predicted_quantity:.0f} units). "
                       f"Consider aggressive discounting or product discontinuation."
            ))
            recommendations.append(Recommendation(
                action="INVESTIGATE CAUSE",
                priority="HIGH",
                impact="Understand decline in demand",
                details="Analyze market conditions, competition, and customer feedback for this product"
            ))
        
        # Price-based recommendations
        if self.unit_price > self.avg_unit_price * 1.2:
            recommendations.append(Recommendation(
                action="REVIEW PRICING STRATEGY",
                priority="MEDIUM",
                impact="Align pricing with market",
                details=f"Current price (£{self.unit_price:.2f}) is {((self.unit_price/self.avg_unit_price - 1) * 100):.0f}% above average. "
                       f"May impact demand."
            ))
        
        # Financial recommendations
        if financials['profit_margin_pct'] < 10:
            recommendations.append(Recommendation(
                action="IMPROVE MARGINS",
                priority="MEDIUM",
                impact="Increase profitability",
                details=f"Profit margin is low at {financials['profit_margin_pct']:.1f}%. "
                       f"Explore cost reduction or price optimization."
            ))
        
        return recommendations
    
    def generate_alerts(self) -> List[Alert]:
        """Generate alerts for critical situations"""
        alerts = []
        demand_level, confidence = self.classify_demand()
        financials = self.calculate_financial_impact()
        
        # Critical demand alerts
        if demand_level == DemandLevel.VERY_HIGH:
            alerts.append(Alert(
                severity=AlertSeverity.CRITICAL,
                title="CRITICAL: Extreme Demand Spike",
                message=f"Predicted demand is {self.predicted_quantity/self.avg_historical_quantity:.1f}x normal levels!",
                recommendation="Immediately coordinate with procurement and warehouse to increase stock"
            ))
        
        if demand_level == DemandLevel.VERY_LOW:
            alerts.append(Alert(
                severity=AlertSeverity.WARNING,
                title="WARNING: Demand Collapse",
                message=f"Predicted demand at critical low: {self.predicted_quantity:.0f} units ({(self.predicted_quantity/self.avg_historical_quantity * 100):.0f}% of average)",
                recommendation="Review product viability and market conditions"
            ))
        
        # Stockout risk
        if self.predicted_quantity > self.avg_historical_quantity * 1.5 and confidence > 0.80:
            alerts.append(Alert(
                severity=AlertSeverity.CRITICAL,
                title="HIGH STOCKOUT RISK",
                message=f"High confidence prediction of significant demand surge",
                recommendation="Verify inventory levels; prepare expedited restocking plan"
            ))
        
        # Revenue opportunity
        if financials['revenue_variance'] > financials['historical_revenue'] * 0.3:
            alerts.append(Alert(
                severity=AlertSeverity.SUCCESS,
                title="REVENUE OPPORTUNITY",
                message=f"Potential revenue increase of £{financials['revenue_variance']:.2f} ({financials['revenue_variance_pct']:.1f}%)",
                recommendation="Ensure operational readiness to capture this opportunity"
            ))
        
        # Profit concerns
        if financials['estimated_profit'] < 0:
            alerts.append(Alert(
                severity=AlertSeverity.WARNING,
                title="PROFITABILITY CONCERN",
                message=f"Estimated profit is negative: £{financials['estimated_profit']:.2f}",
                recommendation="Review pricing strategy or reduce holding costs"
            ))
        
        # Low confidence alert
        if confidence < 0.65:
            alerts.append(Alert(
                severity=AlertSeverity.WARNING,
                title="LOW PREDICTION CONFIDENCE",
                message=f"Prediction confidence is {confidence*100:.0f}% - results may be unreliable",
                recommendation="Consider manual review or use ensemble predictions"
            ))
        
        return alerts
    
    def get_inventory_recommendation(self) -> Dict:
        """Get specific inventory recommendations"""
        demand_level, confidence = self.classify_demand()
        
        if self.avg_historical_quantity == 0:
            safety_stock = self.predicted_quantity * 0.2
            reorder_point = self.predicted_quantity * 1.2
            max_stock = self.predicted_quantity * 1.5
        else:
            # ABC Analysis-like approach
            demand_ratio = self.predicted_quantity / self.avg_historical_quantity
            safety_stock = self.predicted_quantity * 0.15 * (1 + max(0, demand_ratio - 1))
            reorder_point = self.predicted_quantity + safety_stock
            max_stock = self.predicted_quantity * 2.0
        
        return {
            'safety_stock': max(0, safety_stock),
            'reorder_point': max(0, reorder_point),
            'max_stock_level': max_stock,
            'recommended_order_qty': max(0, self.predicted_quantity * 1.1),
            'lead_time_days': 7,
            'estimated_stock_turnover': self.predicted_quantity / max(1, self.avg_historical_quantity)
        }
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary of analysis"""
        demand_level, confidence = self.classify_demand()
        financials = self.calculate_financial_impact()
        recommendations = self.generate_recommendations()
        alerts = self.generate_alerts()
        inventory = self.get_inventory_recommendation()
        
        return {
            'product_code': self.stock_code,
            'country': self.country,
            'unit_price': self.unit_price,
            'predicted_quantity': self.predicted_quantity,
            'historical_average': self.avg_historical_quantity,
            'demand_level': demand_level.value,
            'confidence': confidence,
            'financials': financials,
            'recommendations': recommendations,
            'alerts': alerts,
            'inventory': inventory
        }


def calculate_aggregate_metrics(predictions_df: pd.DataFrame) -> Dict:
    """
    Calculate aggregate metrics for multiple predictions
    
    Args:
        predictions_df: DataFrame with columns: unit_price, predicted_quantity, historical_avg
    
    Returns:
        Dictionary with aggregate metrics
    """
    total_predicted_revenue = (predictions_df['unit_price'] * predictions_df['predicted_quantity']).sum()
    total_historical_revenue = (predictions_df['unit_price'] * predictions_df['historical_avg']).sum()
    
    avg_confidence = 0.80  # Placeholder
    total_alerts_critical = 0
    
    return {
        'total_predicted_revenue': total_predicted_revenue,
        'total_historical_revenue': total_historical_revenue,
        'revenue_change': total_predicted_revenue - total_historical_revenue,
        'revenue_change_pct': ((total_predicted_revenue - total_historical_revenue) / total_historical_revenue * 100) if total_historical_revenue > 0 else 0,
        'avg_confidence': avg_confidence,
        'total_products_analyzed': len(predictions_df),
        'high_demand_products': len(predictions_df[predictions_df['predicted_quantity'] > predictions_df['historical_avg'] * 1.5]),
        'low_demand_products': len(predictions_df[predictions_df['predicted_quantity'] < predictions_df['historical_avg'] * 0.5])
    }
