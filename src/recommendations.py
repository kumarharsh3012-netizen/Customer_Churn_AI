import pandas as pd
# Generate retention recommendations

def generate_recommendations(customer):

    recommendations = []

    # Contract-based recommendation
    if customer["Contract"] == "Month-to-month":
        recommendations.append(
            "Offer a discount for switching to a longer-term contract."
        )

    # Tenure-based recommendation
    if customer["tenure"] < 12:
        recommendations.append(
            "Provide an onboarding benefit to improve customer engagement."
        )

    # Monthly charges recommendation
    if customer["MonthlyCharges"] > 80:
        recommendations.append(
            "Review pricing and offer a suitable cost-saving plan."
        )

    # Technical support recommendation
    if customer["TechSupport"] == "No":
        recommendations.append(
            "Offer a technical support package or assistance."
        )

    # Online security recommendation
    if customer["OnlineSecurity"] == "No":
        recommendations.append(
            "Offer information about available online security services."
        )

    # Default recommendation
    if not recommendations:
        recommendations.append(
            "Maintain engagement through regular customer satisfaction check-ins."
        )

    return recommendations
# Add recommendations to customer prediction results

def add_recommendations(prediction_results):

    results = prediction_results.copy()

    results["Retention Recommendations"] = results.apply(
        lambda customer: (
            generate_recommendations(customer)
            if customer["Churn Prediction"] == "Yes"
            else ["No immediate retention action required."]
        ),
        axis=1
    )

    return results