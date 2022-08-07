import unittest

import numpy as np
import pandas as pd
from tensorflow import keras

from ryan_adams.layers import BaseComponentLayer, Seasonality, LinearTrend


class TestBaseComponentLayer(unittest.TestCase):
    def test___init_parameter_single_dimension(self):
        t = keras.layers.Input((1,), name='t')
        id = keras.layers.Input((1,), name='id')

        class A(BaseComponentLayer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.a = self._init_parameter(10, 'a')
                
            def call(self, inputs):
                return self.a(inputs[1])

        m = keras.models.Model(inputs=[t, id], outputs=A(n_items=1)([t, id]))
        component_output = m.predict(
            pd.DataFrame({'t': 0, 'id': [0]}).to_dict('series')
        )

        actual = component_output.shape
        expected = (1, 10)
        self.assertEqual(actual, expected)

    def test___init_parameter_multi_dimensional(self):
        t = keras.layers.Input((1,), name='t')
        id = keras.layers.Input((1,), name='id')

        class A(BaseComponentLayer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.b = self._init_parameter((10, 10), 'b')
                
            def call(self, inputs):
                return self.b(inputs[1])

        m = keras.models.Model(inputs=[t, id], outputs=A(n_items=1)([t, id]))
        component_output = m.predict(
            pd.DataFrame({'t': 0, 'id': [0]}).to_dict('series')
        )

        actual = component_output.shape
        expected = (1, 10, 10)
        self.assertEqual(actual, expected)

    def test___init_parameter_multiple_parameters(self):
        t = keras.layers.Input((1,), name='t')
        id = keras.layers.Input((1,), name='id')

        class A(BaseComponentLayer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.a = self._init_parameter(10, 'a')
                self.b = self._init_parameter((10, 10), 'b')
                
            def call(self, inputs):
                return self.a(inputs[1]) + self.b(inputs[1])

        m = keras.models.Model(inputs=[t, id], outputs=A(n_items=1)([t, id]))
        component_output = m.predict(
            pd.DataFrame({'t': 0, 'id': [0]}).to_dict('series')
        )

        actual = component_output.shape
        actual = component_output.shape
        expected = (1, 10, 10)
        self.assertEqual(actual, expected)

    def test_keras_compatability_layer_tracking(self):
        class A(BaseComponentLayer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.a = self._init_parameter(10, 'a')
                
            def call(self, inputs):
                return self.a(inputs[1])

        class AB(BaseComponentLayer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.a = self._init_parameter(10, 'a')
                self.b = self._init_parameter((10, 10), 'b')
                
            def call(self, inputs):
                return self.a(inputs[1]) + self.b(inputs[1])

        t = keras.layers.Input((1,), name='t')
        id = keras.layers.Input((1,), name='id')
        layer_a = A(n_items=10)
        layer_ab = AB(n_items=10)
        # calling the layer creates the weights, see
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/
        _ = [L([t, id]) for L in [layer_a, layer_ab]]

        self.assertEqual(len(layer_a.weights), 1)
        self.assertEqual(len(layer_ab.weights), 2)

        self.assertEqual(layer_a.weights[0].numpy().shape, (10, 10,))
        self.assertEqual(layer_ab.weights[0].numpy().shape, (10, 10,))
        self.assertEqual(layer_ab.weights[1].numpy().shape, (10, 10 * 10,))

        self.assertTrue(layer_a.trainable)
        self.assertTrue(layer_ab.trainable)


class TestSeasonality(unittest.TestCase):
    def test_n_items_equal_1(self):
        t = keras.layers.Input((1,), name='t')
        id = keras.layers.Input((1,), name='id')

        seasonality = Seasonality(period=365.25, order=10, n_items=1)([t, id])
        m = keras.models.Model(inputs=[t, id], outputs=seasonality)
        seasonality = m.predict(
            pd.DataFrame({'t': [0, 0], 'id': [0, 0]}).to_dict('series')
        )
        self.assertEqual(seasonality.shape, (2, 1))
        self.assertTrue(seasonality[0, 0] == seasonality[1, 0])

    def test_n_items_greater_than_1(self):
        t = keras.layers.Input((1,), name='t')
        id = keras.layers.Input((1,), name='id')

        seasonality = Seasonality(period=365.25, order=10, n_items=10)([t, id])
        m = keras.models.Model(inputs=[t, id], outputs=seasonality)
        seasonality = m.predict(
            pd.DataFrame({'t': 0, 'id': np.arange(10)}).to_dict('series')
        )
        self.assertEqual(seasonality.shape, (10, 1))
        self.assertNotEqual(seasonality.std(), 0)


class TestLinearTrend(unittest.TestCase):
    def test_dimensionality(self):
        t = keras.layers.Input((1,), name='t')
        id = keras.layers.Input((1,), name='id')

        trend = LinearTrend(n_items=1, t_range=(0, 1))([t, id])
        m = keras.models.Model(inputs=[t, id], outputs=trend)
        trend = m.predict(
            pd.DataFrame({'t': [0, 0], 'id': [0, 0]}).to_dict('series')
        )
        self.assertEqual(trend.shape, (2, 1))
        self.assertTrue(trend[0, 0] == trend[1, 0])

    def test_n_items_greater_than_1(self):
        t = keras.layers.Input((1,), name='t')
        id = keras.layers.Input((1,), name='id')

        trend = Seasonality(period=365.25, order=10, n_items=10)([t, id])
        m = keras.models.Model(inputs=[t, id], outputs=trend)
        trend = m.predict(
            pd.DataFrame({'t': 0, 'id': np.arange(10)}).to_dict('series')
        )
        self.assertEqual(trend.shape, (10, 1))
        self.assertNotEqual(trend.std(), 0)



if __name__ == '__main__':
    unittest.main()
