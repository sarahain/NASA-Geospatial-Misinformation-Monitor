import evidence
import firms
import ner
import verify


def test_haversine_zero_distance():
    assert verify.haversine(34.0, -118.0, 34.0, -118.0) == 0.0


def test_haversine_one_degree_longitude():
    km = verify.haversine(0.0, 0.0, 0.0, 1.0)
    assert abs(km - 111.19) < 0.5


def test_build_bbox():
    assert firms.build_bbox(34.0, -118.0) == "-118.25,33.75,-117.75,34.25"


def test_evaluate_match_labels():
    assert ner.evaluate_match("Los Angeles", "Los Angeles") == "correct"
    assert ner.evaluate_match("Malibu, California", "Malibu") == "partial"
    assert ner.evaluate_match("Canada", "the East Coast") == "incorrect"
    assert ner.evaluate_match("Maui, Hawaii", "") == "missing"


def test_verify_supported_when_detection_is_close():
    detections = [{"latitude": "34.01", "longitude": "-118.00"}]
    result = verify.verify(34.0, -118.0, detections, threshold_km=25)
    assert result["verdict"] == "SUPPORTED"
    assert result["number_of_detections"] == 1


def test_verify_no_match_when_detection_is_far():
    detections = [{"latitude": "40.0", "longitude": "-74.0"}]
    result = verify.verify(34.0, -118.0, detections, threshold_km=25)
    assert result["verdict"] == "NO_MATCH"


def test_verify_insufficient_when_no_data():
    assert verify.verify(34.0, -118.0, None)["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_evidence_score_is_bounded_and_monotonic():
    close = evidence._score(2.0, 1.0, 3, 0.8)
    far = evidence._score(45.0, 40.0, 1, 0.8)
    assert 0.0 <= far <= close <= 1.0


def test_build_evidence_flags_ambiguous_location():
    claim = {"claim_id": 99, "text": "fire somewhere", "date": "2023-08-08"}
    resolution = {
        "raw_location": "Somewhere",
        "normalized_location": "Somewhere, Nowhere",
        "latitude": 10.0,
        "longitude": 10.0,
        "location_confidence": 0.1,
    }
    record = evidence.build_evidence(claim, resolution, detections=None)
    assert record["verdict"] == "LOCATION_AMBIGUOUS"


def test_build_evidence_supported_with_nearby_detection():
    claim = {"claim_id": 4, "text": "Large fire reported near Lahaina", "date": "2023-08-08"}
    resolution = {
        "raw_location": "Lahaina",
        "normalized_location": "Lahaina, Maui County, Hawaii, United States",
        "latitude": 20.8739,
        "longitude": -156.6777,
        "location_confidence": 0.65,
    }
    detections = [
        {
            "latitude": "20.88",
            "longitude": "-156.68",
            "acq_date": "2023-08-08",
            "acq_time": "1200",
            "satellite": "N20",
            "instrument": "VIIRS",
            "frp": "15.0",
        }
    ]
    record = evidence.build_evidence(claim, resolution, detections, threshold_km=25)
    assert record["verdict"] == "SUPPORTED"
    assert record["number_of_detections"] == 1
    assert record["nearest_detection_km"] is not None
    assert 0.0 <= record["evidence_score"] <= 1.0
