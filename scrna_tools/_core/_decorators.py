from functools import wraps


def embedding_plot(func):
    from ..plotting import embedding_plot
    
    @wraps(embedding_plot)
    def with_embedding(*args, **kwargs):
        return func(*args, **kwargs)

    return with_embedding
