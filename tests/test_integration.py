import test1
import test2
import test3
import test4
import test5
import test6
import test7
import test8
import test9
import test10
from classes import CacheManager, DiskCache, OpenAiDriver, SqlCache
from hypster import Composer


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
        selections={"model": "gpt-4o-mini"}, #TODO: think of what todo if llm_driver.model is selected
        overrides={"llm_driver.max_tokens": 300}
    )

    assert isinstance(result["llm_driver"], OpenAiDriver)
    assert result["llm_driver"].model == "gpt-4o-mini"
    assert result["llm_driver"].max_tokens == 300

def test_external_variables_different_name():
    config = Composer().with_modules(test4).compose()
    result = config.instantiate(
        final_vars=["llm_driver"],
        selections={"openai_model": "gpt-4o-mini"},
        overrides={"llm_driver.max_tokens": 300}
    )

    assert isinstance(result["llm_driver"], OpenAiDriver)
    assert result["llm_driver"].model == "gpt-4o-mini"
    assert result["llm_driver"].max_tokens == 300
    
def test_external_variables_different_name_propogation():
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
        overrides={"llm_driver.max_tokens": 300, 
                   "tabular_driver.model" : "gpt-4o"}
    )

    assert isinstance(result["llm_driver"], OpenAiDriver)
    assert isinstance(result["tabular_driver"], OpenAiDriver)
    assert result["llm_driver"].model == "gpt-4o-mini"
    assert result["tabular_driver"].model == "gpt-4o"
    assert result["llm_driver"].max_tokens == 300
    assert result["tabular_driver"].max_tokens == 1500

def test_hierarchical():
    config = Composer().with_modules(test5).compose()
    result = config.instantiate(final_vars=["cache_manager"],
                                selections={"cache_manager.cache.path":"/var/tmp"})

    assert isinstance(result["cache_manager"], CacheManager)
    assert isinstance(result["cache_manager"].cache, DiskCache)
    assert result["cache_manager"].cache.path == "/var/tmp"

def test_hierarchical_azure():
    config = Composer().with_modules(test8).compose()
    result = config.instantiate(final_vars=["azure_doc_ai_loader"])
    assert isinstance(result["azure_doc_ai_loader"].cache_manager.cache, DiskCache)
    assert result["azure_doc_ai_loader"].cache_manager.cache.cache_dir == "cache/azure_ai_doc"

def test_non_existing_selections():
    config = Composer().with_modules(test8).compose()
    #assert that there's an error:
    try:
        config.instantiate(final_vars=["azure_doc_ai_loader"], 
                           selections={"azure_doc_ai_loader" : "none"})
    except Exception as e:
        assert "azure_doc_ai_loader" in str(e)    

def test_hierarchical_different_caches():
    config = Composer().with_modules(test6).compose()
    result = config.instantiate(
        final_vars=["cache_manager"],
        selections={"cache_manager.cache": "sql_cache"}
    )

    assert isinstance(result["cache_manager"], CacheManager)
    assert isinstance(result["cache_manager"].cache, SqlCache)
    assert result["cache_manager"].cache.table == "cache"

def test_override_implicit_arg():
    config = Composer().with_modules(test1).compose()
    result = config.instantiate(
        final_vars=["llm_driver"],
        selections={"llm_driver": "openai"},
        overrides={"llm_driver.openai.max_tokens": 200,
                   "llm_driver.openai.model" : "gpt-4o-mini"
                   },
    )
    assert isinstance(result["llm_driver"], OpenAiDriver)
    assert result["llm_driver"].max_tokens == 200
    assert result["llm_driver"].model == "gpt-4o-mini"

def test_tuple_var():
    config = Composer().with_modules(test7).compose()
    result = config.instantiate(
        final_vars=["vectorizer", "top_k"],
    )
    assert result["vectorizer"].ngram_range == (1, 3)

def test_lazy_global_update_single():
    config = Composer().with_modules(test9).compose()
    result = config.instantiate(
        final_vars=["cache"],
        selections={"cache.path": "/tmp"},
    )
    assert isinstance(result["cache"], DiskCache)
    assert result["cache"].path == "/tmp"

def test_lazy_global_update_multiple():
    config = Composer().with_modules(test10).compose()
    result = config.instantiate(
        final_vars=["cache_manager"],
        selections={"cache_manager.cache": "sql_cache"},
    )
    assert isinstance(result["cache_manager"].cache, SqlCache)
    assert result["cache_manager"].cache.table == "cache"
    assert isinstance(result["cache_manager"], CacheManager)
    
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
    test_hierarchical_azure()
    test_hierarchical_different_caches()
    #test_override_implicit_arg()
    #test_non_existing_selections() #TODO: test for selections that are not present
    #builder = (Builder().with_adapters(tqdm)).with_config(conf) #TODO: handle this case, including overriding adapters and configs
    test_tuple_var()
    test_lazy_global_update_single()
    test_lazy_global_update_multiple()
    print("All tests passed!")