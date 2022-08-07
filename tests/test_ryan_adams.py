import unittest
from unittest import mock

import numpy as np
import pandas as pd
from tensorflow import keras

from ryan_adams.ryan_adams import RyanAdams


class TestRyanAdams(unittest.TestCase):
    @unittest.skip('need to implement')
    def test_keras_compatability_layer_tracking(self):
        assert False

    @mock.patch('ryan_adams.RyanAdams.__init__')
    def test__build_component_layers(self, mock___init__):
        t = object()
        id = object()
        mock___init__.return_value = None
        mock_trend1, mock_trend2 = mock.Mock(n_items=1), mock.Mock(n_items=2)
        ryan_adams = RyanAdams()
        ryan_adams._base_inputs = {'t': t, 'id': id}

        ryan_adams._build_component_layers([mock_trend1, mock_trend2])
        mock_trend1.assert_called_with([t, None])
        mock_trend2.assert_called_with([t, id])

    def test__build_component_layer_n_items_greater_than_1(self):
        t = object()
        id = object()
        mock_trend = mock.Mock(n_items=2)
        mock_ryan_adams = mock.Mock(_base_inputs={'t': t, 'id': id})

        RyanAdams._build_component_layer(mock_ryan_adams, mock_trend)
        mock_trend.assert_called_with([t, id])

    def test__build_component_layer_n_items_equal_1(self):
        t = object()
        id = object()
        mock_trend = mock.Mock(n_items=1)
        mock_ryan_adams = mock.Mock(_base_inputs={'t': t, 'id': id})

        RyanAdams._build_component_layer(mock_ryan_adams, mock_trend)
        mock_trend.assert_called_with([t, None])

    def test__build_component_layer_n_items_greater_than_1(self):
        t = object()
        id = object()
        mock_trend = mock.Mock(n_items=2)
        mock_ryan_adams = mock.Mock(_base_inputs={'t': t, 'id': id})

        RyanAdams._build_component_layer(mock_ryan_adams, mock_trend)
        mock_trend.assert_called_with([t, id])

    def test__build_prophet_layer(self):
        t = object()
        id = object()
        mock_trend = mock.Mock(n_items=1)
        mock_ryan_adams = mock.Mock(_base_inputs={'t': t, 'id': id})

        RyanAdams._build_component_layer(mock_ryan_adams, mock_trend)
        mock_trend.assert_called_with([t, None])

    def test__list_if_str_or_none(self):
        actual1 = RyanAdams._list_if_str_or_none('a')
        expected1 = ['a']
        self.assertEqual(actual1, expected1)

        actual2 = RyanAdams._list_if_str_or_none(None)
        expected2 = []
        self.assertEqual(actual2, expected2)

        actual3 = RyanAdams._list_if_str_or_none([1, 2])
        expected3 = [1, 2]
        self.assertEqual(actual3, expected3)

    @mock.patch('ryan_adams.RyanAdams._max_n_items')
    @mock.patch('ryan_adams.RyanAdams._build_model')
    def test__build_inputs_n_items_equal_1(self, mock__build_model, mock__max_n_items):
        mock__max_n_items.return_value = 1
        ryan_adams = RyanAdams()
        base_inputs, feature_inputs = ryan_adams._build_inputs()
        self.assertNotIn('id', ryan_adams._base_inputs)

    @mock.patch('ryan_adams.RyanAdams._max_n_items')
    @mock.patch('ryan_adams.RyanAdams._build_model')
    def test__build_inputs_n_items_greater_than_1(self, mock__build_model, mock__max_n_items):
        mock__max_n_items.return_value = 2
        ryan_adams = RyanAdams()
        base_inputs, feature_inputs = ryan_adams._build_inputs()
        self.assertIn('id', ryan_adams._base_inputs)

    @mock.patch('ryan_adams.RyanAdams._max_n_items')
    @mock.patch('ryan_adams.RyanAdams._build_model')
    def test__build_inputs_user_inputs(self, mock__build_model, mock__max_n_items):
        mock__max_n_items.return_value = 2
        ryan_adams = RyanAdams(inputs={'t': 1, 'id': -1})
        base_inputs, feature_inputs = ryan_adams._build_inputs()
        actual = ryan_adams._base_inputs
        expected = {'t': 1, 'id': -1}
        self.assertEqual(actual, expected)

    def test__get_or_create(self):
        sentinel1 = object()
        sentinel2 = object()
        mock1 = mock.Mock()
        mock1.return_value = sentinel1
        mock2 = mock.Mock()

        actual1 = RyanAdams._get_or_create({}, 'a', mock1, 1, 2, a=1)
        expected1 = sentinel1
        actual2 = RyanAdams._get_or_create({'a': sentinel2}, 'a', mock1, 1, 2, a=1)
        expected2 = sentinel2

        self.assertIs(actual1, expected1)
        self.assertIs(actual2, expected2)
        mock1.assert_called_with(1, 2, a=1)
        mock2.assert_not_called()


if __name__ == '__main__':
    unittest.main()
