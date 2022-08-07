import numpy as np
import pandas as pd
from tensorflow import keras


def plot_components(model, global_t_range=None, t_interval=None):
    import matplotlib.pyplot as plt

    n_plots = len(model.trends) + len(model.seasonalities)
    fig, axs = plt.subplots(n_plots, 1, figsize=(5, 10))

    for ax, component in zip(axs, list(model.trends) + list(model.seasonalities)):
        if global_t_range is None:
            if hasattr(component, 't_range'):
                t_range = component.t_range
            elif hasattr(component, 'period'):
                t_range = (0, int(component.period))
            else:
                raise ValueError
        else:
            t_range = global_t_range    
        # TODO: establish terms: component, prophet layers, etc.?
        plot_component(model, component, t_range, t_interval, ax)

def plot_component(model, component, t_range, t_interval=None, ax=None):
    import pandas as pd

    # infer sensible defaults
    if isinstance(component, str):
        component = model._model.get_layer(component)

    if t_interval is None:
        t_interval = int(t_range[1] - t_range[0] + 1)

    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))

    # generate the outputs
    m = keras.models.Model(inputs=list(model._base_inputs.values()), outputs=model._build_prophet_layers([component]))
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

    # make the plots
    ax.set_title(component.name)
    for _, group in X.groupby('id'):
        ax.plot(group.t, group.prediction)
