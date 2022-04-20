from unittest import TestCase
from jqpd import jqpd
import numpy as np


class TestJQPD(TestCase):
    
    lbound = 0
    ubound = 100
    alpha = 0.05
    low = 15
    base = 30
    high = 45
    
    def test_quantile_b1(self):
        s = jqpd.quantile(self.alpha, self.alpha, self.low, self.base, self.high, self.lbound, self.ubound)
        self.assertTrue(np.round(s, 10) == 15)
        
    def test_quantile_b2(self):
        s = jqpd.quantile(0.5, self.alpha, self.low, self.base, self.high, self.lbound, self.ubound)
        self.assertTrue(np.round(s, 10) == 30)
        
    def test_quantile_b3(self):
        s = jqpd.quantile(1-self.alpha, self.alpha, self.low, self.base, self.high, self.lbound, self.ubound)
        self.assertTrue(np.round(s, 10) == 45)
        
    def test_quantile_s1(self):
        s = jqpd.quantile(self.alpha, self.alpha, self.low, self.base, self.high, self.lbound)
        self.assertTrue(np.round(s, 10) == 15)
        
    def test_quantile_s2(self):
        s = jqpd.quantile(0.5, self.alpha, self.low, self.base, self.high, self.lbound)
        self.assertTrue(np.round(s, 10) == 30)
        
    def test_quantile_s3(self):
        s = jqpd.quantile(1-self.alpha, self.alpha, self.low, self.base, self.high, self.lbound)
        self.assertTrue(np.round(s, 10) == 45)
        
    def test_pdf_b1(self):
        s = jqpd.pdf(self.low, self.alpha, self.low, self.base, self.high, self.lbound, self.ubound)
        self.assertTrue(np.round(s, 5) == 0.01188)
        
    def test_pdf_b2(self):
        s = jqpd.pdf(self.base, self.alpha, self.low, self.base, self.high, self.lbound, self.ubound)
        self.assertTrue(np.round(s, 5) == 0.04339)
        
    def test_pdf_b3(self):
        s = jqpd.pdf(self.high, self.alpha, self.low, self.base, self.high, self.lbound, self.ubound)
        self.assertTrue(np.round(s, 5) == 0.01125)    

    def test_pdf_s1(self):
        s = jqpd.pdf(self.low, self.alpha, self.low, self.base, self.high, self.lbound)
        self.assertTrue(np.round(s, 5) == 0.01415)
        
    def test_pdf_s2(self):
        s = jqpd.pdf(self.base, self.alpha, self.low, self.base, self.high, self.lbound)
        self.assertTrue(np.round(s, 5) == 0.03982)
        
    def test_pdf_s3(self):
        s = jqpd.pdf(self.high, self.alpha, self.low, self.base, self.high, self.lbound)
        self.assertTrue(np.round(s, 5) == 0.01260)
        
    def test_cdf_b1(self):
        s = jqpd.cdf(self.low, self.alpha, self.low, self.base, self.high, self.lbound, self.ubound)
        self.assertTrue(np.round(s, 10) == 0.05)
        
    def test_cdf_b2(self):
        s = jqpd.cdf(self.base, self.alpha, self.low, self.base, self.high, self.lbound, self.ubound)
        self.assertTrue(np.round(s, 10) == 0.5)
        
    def test_cdf_b3(self):
        s = jqpd.cdf(self.high, self.alpha, self.low, self.base, self.high, self.lbound, self.ubound)
        self.assertTrue(np.round(s, 10) == 0.95)    

    def test_cdf_s1(self):
        s = jqpd.cdf(self.low, self.alpha, self.low, self.base, self.high, self.lbound)
        self.assertTrue(np.round(s, 10) == 0.05)
        
    def test_cdf_s2(self):
        s = jqpd.cdf(self.base, self.alpha, self.low, self.base, self.high, self.lbound)
        self.assertTrue(np.round(s, 10) == 0.5)
        
    def test_cdf_s3(self):
        s = jqpd.cdf(self.high, self.alpha, self.low, self.base, self.high, self.lbound)
        self.assertTrue(np.round(s, 10) == 0.95)
    
    def test_mean(self):
        s = jqpd.jmean(self.alpha, self.low, self.base, self.high, self.lbound, self.ubound)
        self.assertTrue(np.round(s, 5) == 30.00643)
        
    def test_sd(self):
        s = jqpd.jsd(self.alpha, self.low, self.base, self.high, self.lbound, self.ubound)
        self.assertTrue(np.round(s, 5) == 9.09692)
        
    def test_var(self):
        s = jqpd.jvar(self.alpha, self.low, self.base, self.high, self.lbound, self.ubound)
        self.assertTrue(np.round(s, 5) == 82.75399)
        
    def test_skew(self):
        s = jqpd.jskew(self.alpha, self.low, self.base, self.high, self.lbound, self.ubound)
        self.assertTrue(np.round(s, 5) == 0.05101)
        
    def test_kur(self):
        s = jqpd.jkur(self.alpha, self.low, self.base, self.high, self.lbound, self.ubound)
        self.assertTrue(np.round(s, 5) == 2.90528)
 
        
if __name__ == '__main__':
    unittest.main()