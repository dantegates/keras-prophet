import numpy as np
import tensorflow as tf
from tensorflow import keras

from .layers import _chain_layers


class RyanAdams:
    def __init__(self, *,
                 trends=None, seasonalities=None,
                 outer_layers=None,
                 output_activations=None,
                 feature_names=None,
                 loss='mse',
                 optimizer='adam',
                 inputs=None):
        self.trends = self._list_if_str_or_none(trends)
        self.seasonalities = self._list_if_str_or_none(seasonalities)
        self.outer_layers = self._list_if_str_or_none(outer_layers)
        self.output_activations = self._list_if_str_or_none(output_activations)
        self.feature_names = self._list_if_str_or_none(feature_names)

        self.loss = loss
        self.optimizer = optimizer

        self._inputs = inputs or {}
        self._base_inputs, self._feature_inputs = self._build_inputs()
        self._model = self._build_model()

    def fit(self, X, y, **kwargs):
        self._model.fit(X, y, **kwargs)
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
        model = keras.models.Model(
            inputs=self._model_inputs,
            outputs=Z
        )
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def _build_inputs(self):
        base_inputs = {'t': self._get_or_create(self._inputs, 't', keras.Input, (1,), name='t')}

        if self._max_n_items() > 1:
            base_inputs['id'] = self._get_or_create(self._inputs, 'id', keras.Input, (1,), name='id')

        feature_inputs = {
            f: self._get_or_create(self._inputs, f, keras.Input, (1,), name=f)
            for f in self.feature_names
        }

        return base_inputs, feature_inputs

    def _build_feature_layer(self):
        if self._feature_inputs:
            return self._build_dense_features(self._feature_inputs)
        return None

    def _build_trend_layers(self):
        return self._build_component_layers(self.trends)

    def _build_seasonal_layers(self):
        return self._build_component_layers(self.seasonalities)

    def _build_component_layers(self, layers):
        return [self._build_component_layer(L) for L in layers]

    def _build_component_layer(self, layer):
        t = self._base_inputs['t']
        if layer.n_items == 1:
            id = None
        else:
            id = self._base_inputs['id']
        return layer([t, id])

    def _build_outputs(self, W):
        if self.outer_layers:
            W = _chain_layers(W, *self.outer_layers)

        Z = []
        if self.output_activations:
            for a in self.output_activations:
                Z.append(keras.layers.Dense(1, activation=a)(W))
        else:
            Z.append(keras.layers.Dense(1, kernel_initializer='ones', bias_initializer='zeros', trainable=False)(W))

        return keras.layers.Concatenate()(Z)

    @staticmethod
    def _build_dense_features(features):
        return keras.layers.DenseFeatures([
            tf.feature_column.numeric_column(name)
            for name in features.keys()
        ])(list(features.values()))

    def _max_n_items(self):
        return max(L.n_items for L in self.trends + self.seasonalities)

    def _min_n_items(self):
        return min(L.n_items for L in self.trends + self.seasonalities)

    @staticmethod
    def _list_if_str_or_none(obj):
        if obj is None:
            obj = []
        elif isinstance(obj, str) or not hasattr(obj, '__len__'):
            obj = [obj]
        return obj

    @property
    def _model_inputs(self):
        inputs = [self._base_inputs['t']]
        if self._max_n_items() > 1:
            inputs.append(self._base_inputs['id'])
        inputs.extend(self._feature_inputs.values())
        return inputs

    @staticmethod
    def _get_or_create(mapping, key, fn, *args, **kwargs):
        if key in mapping:
            return mapping[key]
        return fn(*args, **kwargs)
