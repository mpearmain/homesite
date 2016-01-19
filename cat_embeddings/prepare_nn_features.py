import numpy


def split_features(X):
    X = numpy.array(X)
    X_list = []

    store_index = X[..., [1]] - 1
    X_list.append(store_index)

    day_of_week = X[..., [2]] - 1
    X_list.append(day_of_week)

    promo = X[..., [3]]
    X_list.append(promo)

    year = X[..., [4]] - 2013
    X_list.append(year)

    month = X[..., [5]] - 1
    X_list.append(month)

    day = X[..., [6]] - 1
    X_list.append(day)

    state_holiday = X[..., [7]]
    X_list.append(state_holiday)

    school_holiday = X[..., [8]]
    X_list.append(school_holiday)

    has_competition_for_months = X[..., [9]]
    X_list.append(has_competition_for_months)

    has_promo2_for_weeks = X[..., [10]]
    X_list.append(has_promo2_for_weeks)

    latest_promo2_for_months = X[..., [11]]
    X_list.append(latest_promo2_for_months)

    log_distance = X[..., [12]]
    X_list.append(log_distance)

    StoreType = X[..., [13]]
    X_list.append(StoreType)

    Assortment = X[..., [14]]
    X_list.append(Assortment)

    PromoInterval = X[..., [15]]
    X_list.append(PromoInterval)

    Promo2SinceYear = X[..., [17]] - 2008
    Promo2SinceYear[Promo2SinceYear < 0] = 0
    X_list.append(Promo2SinceYear)

    State = X[..., [18]]
    X_list.append(State)

    week_of_year = X[..., [19]] - 1
    X_list.append(week_of_year)

    temperature = X[..., [20, 21, 22]]
    X_list.append(temperature)

    humidity = X[..., [23, 24, 25]]
    X_list.append(humidity)

    wind = X[..., [26, 27]]
    X_list.append(wind)

    cloud = X[..., [28]]
    X_list.append(cloud)

    weather_event = X[..., [29]]
    X_list.append(weather_event)

    return X_list