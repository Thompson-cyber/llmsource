import io
import logging
import pathlib
import tempfile
import unittest
import unittest.mock
import importlib.resources

import numpy as np
import pandas as pd
import pandas.testing as pdtest

from namelist_var import namelist_var
from param_info import param_info
from Models import Model,register_param

"""
Test cases for param_info. Largely generated by chatGPT3 on 2023-04-07 then modified by SFBT
"""


class TestParamInfo(unittest.TestCase):
    def fn(self, model, value):
        return value ** 2

    def setUp(self):
        self.param_info = param_info()
        # set up test case.
        nl_var1 = namelist_var(filepath=pathlib.Path('fred.nl'), namelist='atmos', nl_var='VF1', name='VF1',default=1)
        nl_var2 = namelist_var(filepath=pathlib.Path('fred.nl'), namelist='atmos', nl_var='VF2', name='VF2',default=2.0)
        nl_var3 = namelist_var(filepath=pathlib.Path('fred.nl'), namelist='ocean', nl_var='OCDIFF', name='OCDIFF',default=False)
        self.param_info.register('FN', self.fn)
        self.param_info.register('VF1', nl_var1)
        self.param_info.register('VF1', nl_var2, duplicate=True)
        self.param_info.register('OCDIFF', nl_var3)

        self.expect_dct = dict(
            FN=[['function', self.fn.__qualname__]],
            VF1=[nl_var1, nl_var2],
            OCDIFF=[nl_var3]
        )

        expected_df1 = pd.DataFrame([['VF1', 'namelist_var', 'fred.nl', 'atmos', 'VF1', 'VF1',1],
                                     ['VF1', 'namelist_var', 'fred.nl', 'atmos', 'VF2', 'VF2', 2.0],
                                     ['OCDIFF', 'namelist_var', 'fred.nl', 'ocean', 'OCDIFF', 'OCDIFF', False]],
                                    columns=['parameter', 'type', 'filepath', 'namelist', 'nl_var', 'name', 'default'])
        expected_df2 = pd.DataFrame([['FN', 'function', self.fn.__qualname__]],
                                    columns=['parameter', 'type', 'function_name'])
        self.expected_df = pd.concat([expected_df2, expected_df1], ignore_index=True)
        # reindex so as expected
        cols = list(expected_df1.columns)
        cols2 = list(expected_df2.columns)
        for c in cols:
            try:
                cols2.remove(c)
            except ValueError:
                pass
        cols.extend(cols2)
        self.expected_df = self.expected_df.reindex(columns=cols)
        post_process = dict(script='$OPTCLIMTOP/OptClimVn3/scripts/comp_obs.py', output_file='obs.json')
        self.post_process = post_process

    def test_register(self):
        # Test registered a namelist (well a list of anything)
        p = param_info()
        p.register('VF1', 'nl_var1')
        p.register('VF1', 'nl_var2', duplicate=True)
        self.assertEqual(p.param_constructors['VF1'], ['nl_var1', 'nl_var2'])

        # Test registering a callable
        def set_RHCRIT(model, value):
            model.RHCRIT = value

        p.register('RHCRIT', set_RHCRIT)
        self.assertEqual(p.param_constructors['RHCRIT'], [set_RHCRIT])
        self.assertEqual(p.known_functions[set_RHCRIT.__qualname__], set_RHCRIT)
        self.assertEqual(p.got_vars, {'nl_var1', 'nl_var2', set_RHCRIT})

        # Test overwriting an existing registration
        p.register('VF1', 'new_nl_var1')
        self.assertEqual(p.param_constructors['VF1'], ['new_nl_var1'])
        self.assertEqual(p.got_vars, {set_RHCRIT, 'new_nl_var1'})
        self.assertEqual(p.known_functions[set_RHCRIT.__qualname__], set_RHCRIT)
        # test adding a namelist we already have raises an error.
        with self.assertRaises(ValueError):
            p.register('ENTCOEFF', 'new_nl_var1')

        # test logging works as expected
        def set_RHCRIT2(model, value):
            model.RHCRIT2 = value

        with self.assertLogs(level=logging.DEBUG) as log:
            p.register('VF1', 'new_nl_var5')  # overwrite
            p.register('RHCRIT2', set_RHCRIT2)
        self.assertEqual(log.output, [
            f"INFO:OPTCLIM.param_info:Overwriting VF1 and removing ['new_nl_var1']",
            f"DEBUG:OPTCLIM.param_info:Set VF1 to new_nl_var5",
            f"DEBUG:OPTCLIM.param_info:Parameter RHCRIT2 uses method {set_RHCRIT2.__qualname__} ",
        ])

    def test_param(self):
        # Test with a namelist
        p = param_info()
        nl_var1 = namelist_var(filepath=pathlib.Path('ATMCNTL'), namelist='atmos', nl_var='vf1')
        nl_var2 = namelist_var(filepath=pathlib.Path('ATMCNTL'), namelist='atmos', nl_var='vf2')
        p.register('VF1', nl_var1)
        result = p.param(None, 'VF1', 42)
        self.assertEqual(result, [(nl_var1, 42)])
        p.register('VF1', nl_var2, duplicate=True)
        result = p.param(None, 'VF1', 42)
        self.assertEqual(result, [(nl_var1, 42), (nl_var2, 42)])

        # Test with a callable
        nl_var1a = namelist_var(filepath=pathlib.Path('ATMCNTL'), namelist='atmos', nl_var='rhcrit')
        nl_var2a = namelist_var(filepath=pathlib.Path('ATMCNTL'), namelist='atmos', nl_var='rhcrit2')

        def set_RHCRIT(model, value):
            return [(nl_var1a, [value] * 19), (nl_var2a, [value] * 10)]

        p.register('RHCRIT', set_RHCRIT)
        result = p.param(None, 'RHCRIT', 42)
        self.assertEqual(result, [(nl_var1a, [42] * 19), (nl_var2a, [42] * 10)])
        # test logs
        with self.assertLogs(level=logging.DEBUG) as log:
            a = p.param(None, 'VF1', 42)
            b = p.param(None, 'RHCRIT', 10)
        self.assertEqual(log.output,
                         [f"DEBUG:OPTCLIM.param_info:Parameter VF1 set {nl_var1} to 42",
                          f"DEBUG:OPTCLIM.param_info:Parameter VF1 set {nl_var2} to 42",
                          f"DEBUG:OPTCLIM.param_info:Parameter RHCRIT called {set_RHCRIT.__qualname__} with 10 and returned {[(nl_var1a, [10] * 19), (nl_var2a, [10] * 10)]}"])

        # test failures
        p.param_constructors['fred'] = 2
        with self.assertRaises(ValueError):
            p.param(None, 'fred', 10)
        with self.assertRaises(KeyError):  # key does not exist.
            p.param(None, 'Fred', 10)

        # bad fn.
        def bad_fn(model, value):
            return value

        p.register('bad', bad_fn)
        with self.assertRaises(ValueError):
            p.param(None, 'bad', 10)

    def test_to_dict(self):
        # Test with a namelist
        p = param_info()
        p.register('VF1', 'nl_var1')
        p.register('VF2', 'nl_var2')
        self.assertEqual(p.to_dict(), dict(VF1=['nl_var1'], VF2=['nl_var2']))

        # Test with a callable
        def set_RHCRIT(model, value):
            model.RHCRIT = value

        p.register('RHCRIT', set_RHCRIT)
        self.assertEqual(p.to_dict(),
                         {'VF1': ['nl_var1'], 'VF2': ['nl_var2'],
                          'RHCRIT': [['function', 'TestParamInfo.test_to_dict.<locals>.set_RHCRIT']]})
        p.register('RHCRIT', 'nl_var6', duplicate=True)
        self.assertEqual(p.to_dict(),
                         {'VF1': ['nl_var1'], 'VF2': ['nl_var2'],
                          'RHCRIT': [['function', 'TestParamInfo.test_to_dict.<locals>.set_RHCRIT'], 'nl_var6']})

        # test with std setup value
        dct = self.param_info.to_dict()
        self.assertEqual(dct, self.expect_dct)

    def test_from_dict(self):
        # Test with a namelist
        dct = {'VF1': ['nl_var1', 'nl_var2']}
        result = param_info.from_dict(dct)
        self.assertEqual(result.param_constructors, {'VF1': ['nl_var1', 'nl_var2']})

        # test fn doesn't get added.
        dct = {'VF1': ['nl_var1', 'nl_var2'], 'RHCRIT': [['function', 'set_RHCRIT']]}
        result = param_info.from_dict(dct)
        self.assertEqual(result.param_constructors, {'VF1': ['nl_var1', 'nl_var2']})

    def test_to_DataFrame(self):
        # test we can make a dataframe

        self.param_info.print_parameters()
        df = self.param_info.to_DataFrame()

        pdtest.assert_frame_equal(df, self.expected_df)

    def test_print_parameters(self):
        # test print_parameters works
        @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
        def assert_stdout(self, method, expected_output, mock_stdout):
            method()
            self.assertEqual(mock_stdout.getvalue(), expected_output)

        nl_var1 = namelist_var(filepath=pathlib.Path('fred.nl'), namelist='atmos', nl_var='VF1', name='VF1')
        nl_var2 = namelist_var(filepath=pathlib.Path('fred.nl'), namelist='atmos', nl_var='VF2', name='VF2')
        nl_var3 = namelist_var(filepath=pathlib.Path('fred.nl'), namelist='ocean', nl_var='OCDIFF', name='OCDIFF')

        def fn(model, value):
            return value ** 2

        dct = {'VF1': [nl_var1, nl_var2], 'OCDIFF': [nl_var3], 'RHCRIT': [['function', 'set_RHCRIT']]}
        params = param_info.from_dict(dct)
        params.register('FN', fn)
        expect = f"""VF1 [{nl_var1} {nl_var2} ]
OCDIFF [{nl_var3} ]
FN [function: {fn.__qualname__} ]
"""
        assert_stdout(self, params.print_parameters, expect)

    def test_update_from_file(self):
        # test add_parameters_from_file
        # need to generate a file.
        p = param_info()
        # register functions
        p.register('FN', self.fn)
        with tempfile.NamedTemporaryFile(mode='w', prefix='.csv', delete=False) as tfile:
            file = pathlib.Path(tfile.name)
            self.expected_df.to_csv(file, index=False)
            tfile.close()
            p.update_from_file(file, duplicate=True)
        # now to test have what we expect.
        self.assertEqual(p.to_dict(), self.param_info.to_dict())
        self.assertEqual(vars(p), vars(self.param_info))
        # try again with no fn. Expect a warning message
        p = param_info()
        with tempfile.NamedTemporaryFile(mode='w', prefix='.csv', delete=False) as tfile:
            file = tfile.name
            self.expected_df.to_csv(file, index=False)
            tfile.close()
            with self.assertLogs(level=logging.WARNING) as log:
                p.update_from_file(file, duplicate=True)
            self.assertEqual(log.output, ["WARNING:OPTCLIM.param_info:Function TestParamInfo.fn not found. Likely some discrepancy"])

    def test_update(self):
        """
        Test update method
        :return: nada
        """
        # simple test.
        orig = param_info()
        new = param_info()
        orig.update(new)
        self.assertTrue(len(orig.param_constructors) == 0)  # update empty from empty gives empty
        new.register('cf1', 'nl_cf1')
        self.assertTrue(len(orig.param_constructors) == 0)  # changing new did not change orig
        orig.update(new)  # update empty should have identical
        self.assertEqual(vars(orig), vars(new))
        orig.register('cf2', 'nl_cf2')  # put something into orig
        orig.update(new)  # update it from new.
        new.register('cf2', 'nl_cf2')  # do same for new
        self.assertEqual(vars(orig), vars(new))  # should be identical

    def test_read_param(self):
        """
        Test that read_param works.
        Need to create a model instance and then use that!
        :return:
        """

        class myModel(Model):
            @register_param('RHCRIT')
            def rhcrit(self, rhcrit):
                """
                Compute rhcrit on multiple model levels
                :param rhcrit: meta parameter for rhcrit
                :param inverse: default False. If True invert the relationship
                :return: (value of meta parameter if inverse set otherwise
                   a tuple with namelist_var infor and  a list of rh_crit on model levels

                """
                # Check have 19 levels.
                rhcrit_nl = namelist_var(filepath=pathlib.Path('CNTLATM'), nl_var='RHCRIT', namelist='RUNCNST')
                curr_rhcrit = rhcrit_nl.read_value(dirpath=self.model_dir)
                if len(curr_rhcrit) != 19:
                    raise ValueError("Expect 19 levels")
                inverse = rhcrit is None
                if inverse:
                    return rhcrit_nl.read_value(dirpath=self.model_dir)[3]
                else:
                    cloud_rh_crit = 19 * [rhcrit]
                    cloud_rh_crit[0] = max(0.95, rhcrit)
                    cloud_rh_crit[1] = max(0.9, rhcrit)
                    cloud_rh_crit[2] = max(0.85, rhcrit)
                    return (rhcrit_nl, cloud_rh_crit)

        traverse = importlib.resources.files("Models")
        #myModel.remove_param() # clean it up!
        #myModel.register_functions()
        with importlib.resources.as_file(traverse.joinpath("parameter_config/example_Parameters.csv")) as pth:
            myModel.update_from_file(pth)

        with tempfile.TemporaryDirectory() as tmpdir:
            p=pathlib.Path(tmpdir)
            model = myModel('fred',myModel.expand("$OPTCLIMTOP/OptClimVn3/configurations/example_Model/reference")
                                                  ,self.post_process,  model_dir=p)  # depends on myModel
            model.instantiate()
            self.assertEqual(model.param_info.read_param(model, 'VF1'), 1)
            self.assertEqual(model.param_info.read_param(model, 'RHCRIT'), 0.7)


if __name__ == '__main__':
    unittest.main()
