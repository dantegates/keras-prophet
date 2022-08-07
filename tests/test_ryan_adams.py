import unittest
from unittest import mock

import numpy as np
import pandas as pd
from tensorflow import keras

from ryan_adams.ryan_adams import RyanAdams


class TestRyanAdams(unittest.TestCase):
    def test_keras_compatability_layer_tracking(self):
        assert False

    def test__build_component_layers(self):
        t = object()
        id = object()
        mock_trend1, mock_trend2 = mock.Mock(n_items=1), mock.Mock(n_items=2)
        mock_ryan_adams = mock.Mock(_base_inputs={'t': t, 'id': id})
        mock_ryan_adams._build_component_layer = RyanAdams._build_component_layer

        RyanAdams._build_component_layers(mock_ryan_adams, [mock_trend1, mock_trend2])
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


if __name__ == '__main__':
    unittest.main()
