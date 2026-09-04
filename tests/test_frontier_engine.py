"""
Tests for FrontierDomainEngine threshold evaluation and security features.
"""
import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from diff_privacy_engine.engine import FrontierDomainEngine


class TestFrontierDomainEnginePrimary(unittest.TestCase):
    """Tests for evaluate_primary_parameter."""

    def test_nominal_value_returns_none(self):
        result = FrontierDomainEngine.evaluate_primary_parameter(10.0)
        self.assertIsNone(result)

    def test_at_threshold_returns_none(self):
        result = FrontierDomainEngine.evaluate_primary_parameter(25.0)
        self.assertIsNone(result)

    def test_above_threshold_returns_alert(self):
        result = FrontierDomainEngine.evaluate_primary_parameter(35.0)
        self.assertIsNotNone(result)
        self.assertIn("summary", result)
        self.assertIn("details", result)
        self.assertIn("remediation", result)

    def test_alert_contains_value(self):
        result = FrontierDomainEngine.evaluate_primary_parameter(42.5)
        self.assertIn("42.50", result["details"])

    def test_zero_returns_none(self):
        result = FrontierDomainEngine.evaluate_primary_parameter(0.0)
        self.assertIsNone(result)

    def test_negative_returns_none(self):
        result = FrontierDomainEngine.evaluate_primary_parameter(-5.0)
        self.assertIsNone(result)


class TestFrontierDomainEngineSecondary(unittest.TestCase):
    """Tests for evaluate_secondary_kinetics."""

    def test_nominal_returns_none(self):
        result = FrontierDomainEngine.evaluate_secondary_kinetics(5.0, False)
        self.assertIsNone(result)

    def test_at_threshold_returns_none(self):
        result = FrontierDomainEngine.evaluate_secondary_kinetics(12.0, False)
        self.assertIsNone(result)

    def test_above_threshold_returns_alert(self):
        result = FrontierDomainEngine.evaluate_secondary_kinetics(15.0, False)
        self.assertIsNotNone(result)
        self.assertIn("summary", result)

    def test_critical_flag_triggers_alert(self):
        result = FrontierDomainEngine.evaluate_secondary_kinetics(5.0, True)
        self.assertIsNotNone(result)
        self.assertIn("CriticalFlag=True", result["details"])

    def test_both_critical_and_high(self):
        result = FrontierDomainEngine.evaluate_secondary_kinetics(20.0, True)
        self.assertIsNotNone(result)


class TestFrontierDomainEngineConformance(unittest.TestCase):
    """Tests for audit_specification_conformance."""

    def test_nominal_returns_none(self):
        result = FrontierDomainEngine.audit_specification_conformance("NOMINAL", {})
        self.assertIsNone(result)

    def test_discordant_returns_alert(self):
        result = FrontierDomainEngine.audit_specification_conformance("DISCORDANT_ANOMALY", {})
        self.assertIsNotNone(result)
        self.assertIn("summary", result)

    def test_anomaly_keyword(self):
        result = FrontierDomainEngine.audit_specification_conformance("ANOMALY", {})
        self.assertIsNotNone(result)

    def test_mutant_keyword(self):
        result = FrontierDomainEngine.audit_specification_conformance("MUTANT_VARIANT", {})
        self.assertIsNotNone(result)

    def test_fail_keyword(self):
        result = FrontierDomainEngine.audit_specification_conformance("FAIL", {})
        self.assertIsNotNone(result)

    def test_case_insensitive(self):
        result = FrontierDomainEngine.audit_specification_conformance("discordant", {})
        self.assertIsNotNone(result)

    def test_empty_descriptor_returns_none(self):
        result = FrontierDomainEngine.audit_specification_conformance("", {})
        self.assertIsNone(result)


class TestPHIGuard(unittest.TestCase):
    """Tests for PHI outbound guard security."""

    def test_clean_text_passes(self):
        from agents.base import PHIGuard
        # Should not raise
        PHIGuard.assert_no_phi("Normal clinical reading within bounds")

    def test_mrn_blocked(self):
        from agents.base import PHIGuard, SecurityException
        with self.assertRaises(SecurityException):
            PHIGuard.assert_no_phi("Patient MRN-12345678 test")

    def test_ssn_blocked(self):
        from agents.base import PHIGuard, SecurityException
        with self.assertRaises(SecurityException):
            PHIGuard.assert_no_phi("SSN: 123-45-6789")

    def test_phone_blocked(self):
        from agents.base import PHIGuard, SecurityException
        with self.assertRaises(SecurityException):
            PHIGuard.assert_no_phi("Call 555-123-4567")

    def test_email_blocked(self):
        from agents.base import PHIGuard, SecurityException
        with self.assertRaises(SecurityException):
            PHIGuard.assert_no_phi("Email patient@example.com")

    def test_none_input_safe(self):
        from agents.base import PHIGuard
        PHIGuard.assert_no_phi(None)

    def test_redact_phi(self):
        from agents.base import PHIGuard
        result = PHIGuard.redact_phi("Patient MRN-12345678 has SSN 123-45-6789")
        self.assertNotIn("12345678", result)
        self.assertNotIn("123-45-6789", result)
        self.assertIn("[REDACTED_IDENTIFIER]", result)


class TestAuditTrail(unittest.TestCase):
    """Tests for HMAC-SHA256 audit trail security."""

    def test_ephemeral_key_warning(self):
        import os
        # Ensure no env key set
        old_key = os.environ.pop("AUDIT_SECRET_KEY", None)
        try:
            from agents.base import AuditTrail
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                trail = AuditTrail()
                self.assertTrue(any("AUDIT_SECRET_KEY" in str(warning.message) for warning in w))
        finally:
            if old_key:
                os.environ["AUDIT_SECRET_KEY"] = old_key

    def test_custom_key_no_warning(self):
        from agents.base import AuditTrail
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trail = AuditTrail(secret_key="test-key-for-unit-tests")
            self.assertFalse(any("AUDIT_SECRET_KEY" in str(warning.message) for warning in w))

    def test_audit_integrity(self):
        from agents.base import AuditTrail
        trail = AuditTrail(secret_key="test-key")
        trail.log("test_actor", "test_tier", "TEST_EVENT", {"data": "value1"})
        trail.log("test_actor", "test_tier", "TEST_EVENT", {"data": "value2"})
        self.assertTrue(trail.verify_integrity())
        self.assertEqual(len(trail.get_trail()), 2)


if __name__ == "__main__":
    unittest.main()
