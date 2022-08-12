import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from tensorflow import keras


def plot_model_components(model, global_t_range=None, t_interval=None, t0=None, t_units='days', subplots_kws=None):
    n_plots = len(model.trends) + len(model.seasonalities)

    kws = dict(figsize=(5, 10))
    kws.update(subplots_kws or {})
    fig, axs = plt.subplots(n_plots, 1, **kws)

    for ax, component in zip(axs, list(model.trends) + list(model.seasonalities)):
        if global_t_range is None:
            if hasattr(component, 't_range'):
                t_range = component.t_range
            elif hasattr(component, 'period'):
                t_range = (0, int(component.period) - 1)
            else:
                raise ValueError
        else:
            t_range = global_t_range    
        # TODO: establish terms: component, prophet layers, etc.?
        plot_model_component(model, component, t_range, t_interval, t0, t_units, ax)

def plot_model_component(model, component, t_range, t_interval=None, t0=None, t_units='days', ax=None):
    

    # infer sensible defaults
    if isinstance(component, str):
        component = model._model.get_layer(component)

    if t_interval is None:
        t_interval = int(t_range[1] - t_range[0] + 1)

    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))

    # generate the outputs
    m = keras.models.Model(
        inputs=list(model._base_inputs.values()),
        outputs=model._build_component_layer(component))
    X = pd.DataFrame({
        't': np.tile(
            np.linspace(*t_range, num=t_interval + 1),
            component.n_items
        ),
        'id': np.repeat(
            np.arange(component.n_items),
            t_interval + 1
        )
    })
    X['prediction'] = m.predict(X.to_dict('series'))

    ax.set_title(component.name)
    for _, group in X.groupby('id'):
        # make the plots
        if t0 is not None:
            x, formatter = _make_date_index(t0, group.t, t_units)
            formatter = mdates.DateFormatter(formatter)
        else:
            x, formatter = group.t, '{x}'
        ax.plot(x, group.prediction)

    ax.xaxis.set_major_formatter(formatter)


def _make_date_index(t0, t_offset, t_units):
    x = pd.to_datetime(t0) + pd.to_timedelta(t_offset, unit=t_units)

    date_range = (x.max() - x.min()).days
    if date_range <= 1:
        format_str = '%H:%M:%s'
    elif date_range <= 7:
        format_str = '%A'
    else:
        format_str = '%Y-%m-%d'

    return x, format_str
