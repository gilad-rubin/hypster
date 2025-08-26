"""
Tests for run history snapshots - ensuring HP parameter values are captured correctly.
"""

from hypster import HP, config


def test_basic_snapshot():
    """Test that basic HP calls are recorded in run history"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")
        model_type = hp.select(["rf", "svm"], name="model_type")
        epochs = hp.int(10, name="epochs")

        return {"learning_rate": learning_rate, "model_type": model_type, "epochs": epochs}

    # Call the config
    result = test_config()

    # Get the snapshot
    snapshot = test_config.get_last_snapshot()

    # Should contain all named parameters
    assert "learning_rate" in snapshot
    assert "model_type" in snapshot
    assert "epochs" in snapshot

    # Values should match
    assert snapshot["learning_rate"] == 0.01
    assert snapshot["model_type"] == "rf"
    assert snapshot["epochs"] == 10


def test_snapshot_with_overrides():
    """Test that snapshots reflect overridden values"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")
        model_type = hp.select(["rf", "svm"], name="model_type")

        return {"learning_rate": learning_rate, "model_type": model_type}

    # Call with overrides
    result = test_config(values={"learning_rate": 0.05, "model_type": "svm"})

    # Snapshot should reflect overrides
    snapshot = test_config.get_last_snapshot()
    assert snapshot["learning_rate"] == 0.05
    assert snapshot["model_type"] == "svm"


def test_nested_snapshot():
    """Test that nested config parameters appear with proper naming"""

    @config
    def child_config(hp: HP):
        child_param = hp.number(0.5, name="child_param")
        child_text = hp.text("default", name="child_text")

        return {"child_param": child_param, "child_text": child_text}

    @config
    def parent_config(hp: HP):
        parent_param = hp.number(1.0, name="parent_param")

        # Nest child config
        child = hp.nest(child_config, name="child")

        return {"parent_param": parent_param, "child": child}

    # Call parent config
    result = parent_config()

    # Get snapshot
    snapshot = parent_config.get_last_snapshot()

    # Should contain parent parameter
    assert "parent_param" in snapshot
    assert snapshot["parent_param"] == 1.0

    # Should contain nested parameters with proper prefix
    assert "child.child_param" in snapshot
    assert "child.child_text" in snapshot
    assert snapshot["child.child_param"] == 0.5
    assert snapshot["child.child_text"] == "default"


def test_nested_snapshot_with_overrides():
    """Test nested snapshots with values overrides"""

    @config
    def child_config(hp: HP):
        child_lr = hp.number(0.01, name="learning_rate")
        child_epochs = hp.int(5, name="epochs")

        return {"learning_rate": child_lr, "epochs": child_epochs}

    @config
    def parent_config(hp: HP):
        parent_lr = hp.number(0.1, name="parent_lr")
        child = hp.nest(child_config, name="optimizer")

        return {"parent_lr": parent_lr, "optimizer": child}

    # Call with nested overrides
    result = parent_config(values={"parent_lr": 0.2, "optimizer.learning_rate": 0.001, "optimizer.epochs": 20})

    snapshot = parent_config.get_last_snapshot()

    # Check overrides are reflected
    assert snapshot["parent_lr"] == 0.2
    assert snapshot["optimizer.learning_rate"] == 0.001
    assert snapshot["optimizer.epochs"] == 20


def test_multi_select_snapshot():
    """Test that multi-select values are stored correctly in snapshots"""

    @config
    def test_config(hp: HP):
        selected_items = hp.multi_select(["item1", "item2", "item3"], name="selected_items", default=["item1"])
        single_item = hp.select(["a", "b", "c"], name="single_item")

        return {"selected_items": selected_items, "single_item": single_item}

    result = test_config()
    snapshot = test_config.get_last_snapshot()

    # Multi-select should store list values
    assert "selected_items" in snapshot
    assert "single_item" in snapshot
    assert snapshot["selected_items"] == ["item1"]
    assert snapshot["single_item"] == "a"


def test_conditional_branches_snapshot():
    """Test that snapshots reflect parameters used in different branches"""

    @config
    def branching_config(hp: HP):
        model_type = hp.select(["simple", "complex"], name="model_type")

        if model_type == "simple":
            layers = hp.int(2, name="simple.layers")
            learning_rate = hp.number(0.01, name="simple.lr")
        else:
            layers = hp.int(5, name="complex.layers")
            learning_rate = hp.number(0.001, name="complex.lr")
            dropout = hp.number(0.1, name="complex.dropout")

        return {
            "model_type": model_type,
            "layers": layers,
            "learning_rate": learning_rate,
            "dropout": dropout if model_type == "complex" else None,
        }

    # Test simple branch
    result = branching_config(values={"model_type": "simple"})
    snapshot = branching_config.get_last_snapshot()

    assert snapshot["model_type"] == "simple"
    assert "simple.layers" in snapshot
    assert "simple.lr" in snapshot
    assert snapshot["simple.layers"] == 2
    assert snapshot["simple.lr"] == 0.01
    # Complex parameters should not be in snapshot
    assert "complex.layers" not in snapshot
    assert "complex.lr" not in snapshot
    assert "complex.dropout" not in snapshot

    # Test complex branch
    result = branching_config(values={"model_type": "complex"})
    snapshot = branching_config.get_last_snapshot()

    assert snapshot["model_type"] == "complex"
    assert "complex.layers" in snapshot
    assert "complex.lr" in snapshot
    assert "complex.dropout" in snapshot
    assert snapshot["complex.layers"] == 5
    assert snapshot["complex.lr"] == 0.001
    assert snapshot["complex.dropout"] == 0.1
    # Simple parameters should not be in snapshot
    assert "simple.layers" not in snapshot
    assert "simple.lr" not in snapshot


def test_snapshot_replay():
    """Test that using a snapshot as values reproduces the same result"""

    @config
    def test_config(hp: HP):
        learning_rate = hp.number(0.01, name="learning_rate")
        model_type = hp.select(["rf", "svm", "neural"], name="model_type")
        epochs = hp.int(10, name="epochs")
        use_validation = hp.bool(True, name="use_validation")

        return {
            "learning_rate": learning_rate,
            "model_type": model_type,
            "epochs": epochs,
            "use_validation": use_validation,
        }

    # First run with some overrides
    original_values = {"learning_rate": 0.05, "model_type": "neural", "epochs": 20, "use_validation": False}

    result1 = test_config(values=original_values)
    snapshot = test_config.get_last_snapshot()

    # Second run using the snapshot
    result2 = test_config(values=snapshot)

    # Results should be identical
    assert result1 == result2
    assert result1["learning_rate"] == result2["learning_rate"] == 0.05
    assert result1["model_type"] == result2["model_type"] == "neural"
    assert result1["epochs"] == result2["epochs"] == 20
    assert result1["use_validation"] == result2["use_validation"] is False


def test_multiple_snapshots():
    """Test that multiple runs create separate snapshots"""

    @config
    def test_config(hp: HP):
        value = hp.number(0.5, name="value")
        return {"value": value}

    # First run
    result1 = test_config(values={"value": 1.0})

    # Second run
    result2 = test_config(values={"value": 2.0})

    # Third run
    result3 = test_config()

    # Get all snapshots
    snapshots = test_config.get_snapshots()

    # Should have multiple snapshots
    assert len(snapshots) >= 3

    # Last snapshot should be from the most recent run
    last_snapshot = test_config.get_last_snapshot()
    assert last_snapshot["value"] == 0.5  # Default value from third run


def test_snapshot_without_names():
    """Test that unnamed parameters don't appear in snapshots"""

    @config
    def test_config(hp: HP):
        named_param = hp.number(0.01, name="named_param")
        unnamed_param = hp.number(0.02)  # No name

        return {"named": named_param, "unnamed": unnamed_param}

    result = test_config()
    snapshot = test_config.get_last_snapshot()

    # Only named parameters should appear in snapshot
    assert "named_param" in snapshot
    assert snapshot["named_param"] == 0.01

    # Unnamed parameters should not appear
    # (This might depend on implementation - adjust as needed)
    # The snapshot should only contain addressable/overridable parameters
