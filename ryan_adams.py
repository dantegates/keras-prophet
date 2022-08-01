from itertools import chain

import numpy as np
import tensorflow as tf
from tensorflow import keras



def _chain_layers(*layers):
    L = layers[0]
    for layer in layers[1:]:
        L = layer(L)
    return L


class ItemizedLayer(keras.layers.Layer):
    def __init__(self, *args, n_items, shared_embedding_projection=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_items = n_items
        self.shared_embedding_projection = shared_embedding_projection

    # @_cache_param_creation
    # @functools.lru_cache(128)
    def _init_parameter(self, id, dim, name):
        name = f'{self.name}_{name}'

        flattened_dim = dim if isinstance(dim, int) else np.prod(dim)
        output_dim = (dim,) if isinstance(dim, int) else dim

        layers = []
        if id is not None:
            if self.n_items == 1:
                raise ValueError
            layers.append(keras.layers.Embedding(
                input_dim=self.n_items,
                output_dim=flattened_dim,
                input_length=1,
                name=f'{name}_embedding'
            ))
            if self.shared_embedding_projection:
                layers.append(keras.layers.Dense(flattened_dim, name=f'{name}_dense'))

            layers.append(keras.layers.Reshape(output_dim, name=f'{name}_reshape'))

        else:
            assert False, 'what is the right way to index this?'

        return layers


class LinearTrend(ItemizedLayer):
    def __init__(self,
                 *args,
                 t0: int = None,
                 t_range: tuple = None,
                 n_changepoints: int = 20,
                 changepoint_range: float = .8,
                 changepoint_penalty: float = 1e-3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.t0 = t0
        self.t_range = t_range
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.changepoint_penalty = changepoint_penalty

        self.m = self._init_parameter(id, 1, 'm')
        self.k = self._init_parameter(id, 1, 'k')
        self.δ = self._init_parameter(id, (1, self.n_changepoints), 'delta')

    def call(self, input_tensor):
        t, id = input_tensor
        m = _chain_layers(id, *self.m)
        k = _chain_layers(id, *self.k)
        δ = _chain_layers(id, *self.δ)

        # self.add_loss(keras.regularizers.l1(l1=1e-3)(δ))
        changepoints = tf.linspace(
            self.t_range[0],
            int(self.changepoint_range * self.t_range[1]),
            self.n_changepoints + 1)
        s = tf.cast(changepoints, 'float')[1:]

        return self.trend(m, k, δ, t, s)

    @staticmethod
    def trend(m, k, δ, t, s):
        a_t = tf.convert_to_tensor(t > s)
        A = keras.backend.switch(a_t, tf.ones_like(t), tf.zeros_like(t))
        # cast to float and reshape for compatibility with δ
        A = tf.cast(A, 'float')[:, None, :]

        trend = (tf.reduce_sum(tf.multiply(A, δ), axis=-1) + k) * t

        γ = tf.multiply(-s[None, None, :], δ)
        offset = tf.reduce_sum(tf.multiply(A, γ), axis=-1) + m

        return trend + offset


class SaturatingTrend(LinearTrend):
    def __init__(self, *args, cap, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.cap = cap

    def call(self, inputs):
        linear_trend = super().call(inputs)
        return self.cap / (1 + tf.exp(-linear_trend))


class Seasonality(ItemizedLayer):
    def __init__(self,
                 *args,
                 period: float,
                 order: int,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period
        self.order = order
        self.a_n = self._init_parameter(id, self.order, 'a_n')
        self.b_n = self._init_parameter(id, self.order, 'b_n')

    def call(self, input_tensor):
        t, id = input_tensor
        a_n = _chain_layers(id, *self.a_n)
        b_n = _chain_layers(id, *self.b_n)
        n = (tf.range(self.order, dtype='float') + 1)[None, :]
        return self.fourier_series(a_n, b_n, n, self.period, t)

    @staticmethod
    def fourier_series(a_n, b_n, n, period, t):
        x = (2 * np.pi * tf.multiply(t, n)) / period
        return (
            tf.reduce_sum(tf.multiply(tf.cos(x), a_n), axis=1, keepdims=True)
            + tf.reduce_sum(tf.multiply(tf.sin(x), b_n), axis=1, keepdims=True)
        )


class RyanAdadms:
    def __init__(self, *,
                 trends=None, seasonalities=None,
                 outer_layers=None,
                 output_activations='linear',
                 feature_names=None,
                 loss='mse',
                 optimizer='adam'):
        self.trends = self._list_if_str_or_none(trends)
        self.seasonalities = self._list_if_str_or_none(seasonalities)
        self.outer_layers = self._list_if_str_or_none(outer_layers)
        self.output_activations = self._list_if_str_or_none(output_activations)
        self.feature_names = self._list_if_str_or_none(feature_names)

        self.loss = loss
        self.optimizer = optimizer
        
        self._base_inputs, self._feature_inputs = self._build_inputs()
        self._inputs = {**self._base_inputs, **self._feature_inputs}
        self._model = self._build_model()

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def _build_model(self):
        feature_layer = self._build_feature_layer()
        trend_layers = self._build_trend_layers()
        seasonal_layers = self._build_seasonal_layers()
        W = keras.layers.Concatenate()([
            *trend_layers,
            *seasonal_layers,
            *(feature_layer or [])])
        Z = self._build_outputs(W)
        model = keras.models.Model(inputs=list(self._inputs.values()), outputs=Z)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def _build_inputs(self):
        base_inputs = {'t': keras.Input((1,), name='t')}
        if self._max_n_items() > 1:
            base_inputs['id'] = keras.Input((1,), name='id')
        else:
            base_inputs['id'] = None

        feature_inputs = {
            f: keras.Input((1,), name=f) for f in self.feature_names
        }

        return base_inputs, feature_inputs

    def _build_feature_layer(self):
        if self._feature_inputs:
            return self._build_dense_features(self._feature_inputs)
        return None

    def _build_trend_layers(self):
        return self._build_prophet_layers(self.trends)

    def _build_seasonal_layers(self):
        return self._build_prophet_layers(self.seasonalities)

    def _build_prophet_layers(self, layers):
        return [L(list(self._base_inputs.values())) for L in _flatten([layers])]

    def _build_outputs(self, W):
        if self.outer_layers:
            W = _chain_layers(W, *self.outer_layers)

        Z = []
        for a in self.output_activations:
            Z.append(keras.layers.Dense(1, activation=a)(W))

        return keras.layers.Concatenate()(Z)

    @staticmethod
    def _build_dense_features(features):
        return keras.layers.DenseFeatures([
            tf.feature_column.numeric_column(name)
            for name in features.keys()
        ])(list(features.values()))

    def _max_n_items(self):
        return max(L.n_items for L in self.trends + self.seasonalities)

    @staticmethod
    def _list_if_str_or_none(x):
        if x is None:
            x = []
        elif isinstance(x, str) or not hasattr(x, '__len__'):
            x = [x]
        return x


def _flatten(x):
    return list(chain.from_iterable(x))
