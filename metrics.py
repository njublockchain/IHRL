from recbole.evaluator.metrics import ItemCoverage, TailPercentage


def calc_IC(topk, rec_items, data_num_items):
    return ItemCoverage(
        {
            "topk": topk,
            "metric_decimal_place": 8,
        }
    ).calculate_metric(
        {
            "rec.items": rec_items,
            "data.num_items": data_num_items,
        }
    )
