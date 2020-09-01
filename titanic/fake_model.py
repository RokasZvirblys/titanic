def fake_predict(user_age):
    if user_age >10:
        prediction = 'survived (over ten)'
    else:
        prediction = 'Super survived (under ten)'
    return prediction