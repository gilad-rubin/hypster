import test1
import test2
import test3
import test4
import test5
import test6
import inspect
from hypster import CacheManager, Composer, DiskCache, OpenAiDriver, SqlCache


def test_select_two_classes():
    config = Composer().with_modules(test1).compose()
    result = config.instantiate(
        final_vars=["llm_driver"],
        selections={"llm_driver": "openai"},
        overrides={"llm_driver.openai.max_tokens": 200},
    )
    assert isinstance(result["llm_driver"], OpenAiDriver)
    assert result["llm_driver"].max_tokens == 200

def test_select_class_args():
    config = Composer().with_modules(test2).compose()
    result = config.instantiate(
        final_vars=["llm_driver"],
        selections={"llm_driver.model": "gpt-4o-mini"},
        overrides={"llm_driver.max_tokens": 300}
    )

    assert isinstance(result["llm_driver"], OpenAiDriver)
    assert result["llm_driver"].model == "gpt-4o-mini"
    assert result["llm_driver"].max_tokens == 300

def test_empty_selections():
    config = Composer().with_modules(test2).compose()
    result = config.instantiate(
        final_vars=["llm_driver"],
        selections={},
        overrides={"llm_driver.max_tokens": 300}
    )

    assert isinstance(result["llm_driver"], OpenAiDriver)
    assert result["llm_driver"].model == "gpt-4o"
    assert result["llm_driver"].max_tokens == 300
    
def test_external_variables():
    config = Composer().with_modules(test3).compose()
    result = config.instantiate(
        final_vars=["llm_driver"],
        selections={"llm_driver.model": "gpt-4o-mini"},
        overrides={"llm_driver.max_tokens": 300}
    )

    assert isinstance(result["llm_driver"], OpenAiDriver)
    assert result["llm_driver"].model == "gpt-4o-mini"
    assert result["llm_driver"].max_tokens == 300

def test_external_variables_different_name():
    config = Composer().with_modules(test4).compose()
    result = config.instantiate(
        final_vars=["llm_driver"],
        selections={"llm_driver.model": "gpt-4o-mini"},
        overrides={"llm_driver.max_tokens": 300}
    )

    assert isinstance(result["llm_driver"], OpenAiDriver)
    assert result["llm_driver"].model == "gpt-4o-mini"
    assert result["llm_driver"].max_tokens == 300
    
def test_external_variables_different_name_propogation():
    # Dictionary to store objects and their names
    object_references = {}
    
    # First, gather all variable names and their corresponding objects
    for name, obj in inspect.getmembers(test4):
        if not name.startswith("__"):
            object_references[id(obj)] = name

    # Now, inspect class instances and identify their attributes
    for name, obj in inspect.getmembers(test4):
        if not name.startswith("__") and not inspect.isclass(obj) and not inspect.isfunction(obj):
            if hasattr(obj, "__dict__"):  # This checks if obj is a class instance
                print(f"Inspecting object: {name} ({type(obj)})")
                for attr_name, attr_value in vars(obj).items():
                    reference_name = object_references.get(id(attr_value), None)
                    if reference_name:
                        print(f"  Attribute '{attr_name}' points to reference: '{reference_name}'")
                        print(f"    Value of '{reference_name}': {attr_value}")
                    else:
                        print(f"  Attribute '{attr_name}' value: {attr_value}")
    config = Composer().with_modules(test4).compose()
    result = config.instantiate(
        final_vars=["llm_driver"],
        selections={"openai_model": "gpt-4o-mini"},
        overrides={"llm_driver.max_tokens": 300}
    )

    assert isinstance(result["llm_driver"], OpenAiDriver)
    assert result["llm_driver"].model == "gpt-4o-mini"
    assert result["llm_driver"].max_tokens == 300

def test_downstream_propogation():
    config = Composer().with_modules(test4).compose()
    result = config.instantiate(
        final_vars=["llm_driver", "tabular_driver"],
        selections={"openai_model": "gpt-4o-mini"},
        overrides={"llm_driver.max_tokens": 300}
    )

    assert isinstance(result["llm_driver"], OpenAiDriver)
    assert isinstance(result["tabular_driver"], OpenAiDriver)
    assert result["llm_driver"].model == "gpt-4o-mini"
    assert result["tabular_driver"].model == "gpt-4o-mini"
    assert result["llm_driver"].max_tokens == 300
    assert result["tabular_driver"].max_tokens == 1500

def test_hierarchical():
    config = Composer().with_modules(test6).compose()
    result = config.instantiate(final_vars=["cache_manager"],
                                selections={"cache_manager.cache.path":"/var/tmp"})

    assert isinstance(result["cache_manager"], CacheManager)
    assert isinstance(result["cache_manager"].cache, DiskCache)
    assert result["cache_manager"].cache.path == "/var/tmp"

def test_hierarchical_different_caches():
    config = Composer().with_modules(globals()).compose()
    result = config.instantiate(
        final_vars=["cache_manager"],
        selections={"cache_manager.cache": "sql_cache"}
    )

    assert isinstance(result["cache_manager"], CacheManager)
    assert isinstance(result["cache_manager"].cache, SqlCache)
    assert result["cache_manager"].cache.table == "cache"

if __name__ == "__main__":
    import logging

    from hypster import set_debug_level
    set_debug_level(logging.DEBUG)
    test_select_two_classes()
    test_select_class_args()
    test_empty_selections()
    test_external_variables()
    test_external_variables_different_name()
    test_external_variables_different_name_propogation()
    test_downstream_propogation()
    test_hierarchical()
    test_hierarchical_different_caches()