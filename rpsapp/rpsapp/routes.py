from flask import request, render_template
from flask import current_app as app

from .inference import get_customerscore


@app.route ('/', methods=['GET', 'POST'])
def customer_conversion_predict ():
    # Write the GET Method to get the index file
    if request.method == 'GET':
        return render_template ('index.html')
    # Write the POST Method to post the results file
    if request.method == 'POST':
        int_features = [int (x) for x in request.form.values ()]
        # Get customer conversion score
        expected_number_of_purchase, probability_if_alive, customer_persona_predictions = get_customerscore (
            frequency=int_features[0],
            recency=int_features[1],
            Age=int_features[2],
            Monetary=int_features[3]
        )
        # Render the result template
        return render_template ('result.html', cc_score=expected_number_of_purchase, p_alive=probability_if_alive,
                                c_persona=customer_persona_predictions)
