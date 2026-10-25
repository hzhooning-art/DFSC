from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_observed_host as host


class ObservedHostTests(unittest.TestCase):
    def test_background_services_never_get_clean_timing_qualification(self):
        source={'a_accuracy_matched':[{'median_seconds':.2}]}
        report=host.qualified_shared_report(source,[{'unexpected_python_processes':[{'ProcessId':17}]}])
        self.assertFalse(report['primary_uncontended_runtime_claim_supported'])
        self.assertEqual(report['a_accuracy_matched'],[])
        self.assertEqual(len(report['a_raw_accuracy_matched_costs_not_performance_evidence']),1)

    def test_even_no_detected_background_is_not_continuous_host_certificate(self):
        report=host.qualified_shared_report({'a_accuracy_matched':[]},[])
        self.assertFalse(report['primary_uncontended_runtime_claim_supported'])
        self.assertTrue(report['shared_host_observational_execution'])


if __name__=='__main__': unittest.main()
