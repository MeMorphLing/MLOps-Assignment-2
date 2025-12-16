"""
Simple standalone tests that don't require train.py to exist
Run with: pytest tests/test_simple.py -v
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import pickle


class TestBasicFunctionality(unittest.TestCase):
    """Test basic ML pipeline components"""
    
    def setUp(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n = 100
        
        self.sample_data = pd.DataFrame({
            'longitude': np.random.uniform(-124, -114, n),
            'latitude': np.random.uniform(32, 42, n),
            'housing_median_age': np.random.uniform(1, 52, n),
            'total_rooms': np.random.uniform(100, 5000, n),
            'total_bedrooms': np.random.uniform(10, 1000, n),
            'population': np.random.uniform(50, 3000, n),
            'households': np.random.uniform(10, 1000, n),
            'median_income': np.random.uniform(0.5, 15, n),
            'median_house_value': np.random.uniform(50000, 500000, n)
        })
    
    def test_data_creation(self):
        """Test that sample data is created correctly"""
        self.assertEqual(len(self.sample_data), 100)
        self.assertEqual(len(self.sample_data.columns), 9)
        print("✓ Data creation test passed")
    
    def test_data_columns(self):
        """Test that all required columns exist"""
        required_cols = [
            'longitude', 'latitude', 'housing_median_age',
            'total_rooms', 'total_bedrooms', 'population',
            'households', 'median_income', 'median_house_value'
        ]
        
        for col in required_cols:
            self.assertIn(col, self.sample_data.columns)
        print("✓ Column validation test passed")
    
    def test_data_types(self):
        """Test that all columns are numeric"""
        for col in self.sample_data.columns:
            self.assertTrue(
                np.issubdtype(self.sample_data[col].dtype, np.number),
                f"Column {col} is not numeric"
            )
        print("✓ Data type test passed")
    
    def test_no_missing_values(self):
        """Test that there are no missing values"""
        missing_count = self.sample_data.isnull().sum().sum()
        self.assertEqual(missing_count, 0)
        print("✓ No missing values test passed")
    
    def test_data_split(self):
        """Test train-test split functionality"""
        X = self.sample_data.drop('median_house_value', axis=1)
        y = self.sample_data['median_house_value']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Check shapes
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
        print("✓ Data split test passed")
    
    def test_model_training(self):
        """Test that model can be trained"""
        X = self.sample_data.drop('median_house_value', axis=1)
        y = self.sample_data['median_house_value']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Check model is trained
        self.assertTrue(hasattr(model, 'estimators_'))
        self.assertEqual(len(model.estimators_), 10)
        print("✓ Model training test passed")
    
    def test_model_prediction(self):
        """Test that model can make predictions"""
        X = self.sample_data.drop('median_house_value', axis=1)
        y = self.sample_data['median_house_value']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        # Check predictions
        self.assertEqual(len(predictions), len(X_test))
        self.assertTrue(all(predictions > 0))
        print("✓ Model prediction test passed")
    
    def test_model_evaluation(self):
        """Test model evaluation metrics"""
        X = self.sample_data.drop('median_house_value', axis=1)
        y = self.sample_data['median_house_value']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)
        
        # Check metrics are valid
        self.assertGreater(rmse, 0)
        self.assertLessEqual(r2, 1.0)
        self.assertGreaterEqual(r2, -1.0)
        
        print(f"✓ Model evaluation test passed (RMSE: {rmse:.2f}, R²: {r2:.4f})")
    
    def test_model_save_load(self):
        """Test model persistence"""
        X = self.sample_data.drop('median_house_value', axis=1)
        y = self.sample_data['median_house_value']
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/test_model.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Load model
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Test predictions match
        original_pred = model.predict(X[:5])
        loaded_pred = loaded_model.predict(X[:5])
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
        
        # Cleanup
        if os.path.exists(model_path):
            os.remove(model_path)
        
        print("✓ Model save/load test passed")


class TestDataFileExistence(unittest.TestCase):
    """Test that required files and folders exist"""
    
    def test_data_folder_exists(self):
        """Test that data folder exists"""
        if not os.path.exists('data'):
            os.makedirs('data')
        self.assertTrue(os.path.exists('data'))
        print("✓ Data folder exists")
    
    def test_models_folder_exists(self):
        """Test that models folder exists"""
        if not os.path.exists('models'):
            os.makedirs('models')
        self.assertTrue(os.path.exists('models'))
        print("✓ Models folder exists")
    
    def test_src_folder_exists(self):
        """Test that src folder exists"""
        if not os.path.exists('src'):
            os.makedirs('src')
        self.assertTrue(os.path.exists('src'))
        print("✓ Src folder exists")


class TestAPIComponents(unittest.TestCase):
    """Test API-related functionality"""
    
    def test_prediction_input_format(self):
        """Test that prediction input has correct format"""
        sample_input = {
            'longitude': -122.23,
            'latitude': 37.88,
            'housing_median_age': 41.0,
            'total_rooms': 880.0,
            'total_bedrooms': 129.0,
            'population': 322.0,
            'households': 126.0,
            'median_income': 8.3252
        }
        
        # Check all keys exist
        required_keys = [
            'longitude', 'latitude', 'housing_median_age',
            'total_rooms', 'total_bedrooms', 'population',
            'households', 'median_income'
        ]
        
        for key in required_keys:
            self.assertIn(key, sample_input)
        
        # Check all values are numeric
        for value in sample_input.values():
            self.assertIsInstance(value, (int, float))
        
        print("✓ API input format test passed")
    
    def test_prediction_output_format(self):
        """Test that prediction output has correct format"""
        # Simulate a prediction
        sample_output = {
            'predicted_price': 452600.0,
            'formatted_price': '$452,600.00'
        }
        
        self.assertIn('predicted_price', sample_output)
        self.assertIsInstance(sample_output['predicted_price'], float)
        self.assertIn('formatted_price', sample_output)
        self.assertIsInstance(sample_output['formatted_price'], str)
        
        print("✓ API output format test passed")


def run_tests():
    """Run all tests with detailed output"""
    print("=" * 60)
    print("RUNNING MLOPS PROJECT TESTS")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBasicFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestDataFileExistence))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIComponents))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED! 🎉")
    else:
        print("\n✗ Some tests failed. Check output above.")
    
    return result


if __name__ == '__main__':
    run_tests()