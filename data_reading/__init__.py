
from data_reading.KQ_read import NewsKQProcessor
processors = {
    'news_query':NewsKQProcessor,
}


def abstract2sents_func(params):
    task_name = params.task_name.lower()
    if task_name not in processors:
        raise ValueError('Task not found: %s' % task_name)
    return processors[task_name].abstract2sents
