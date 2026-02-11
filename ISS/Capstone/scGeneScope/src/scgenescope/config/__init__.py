from omegaconf import OmegaConf


OmegaConf.register_new_resolver(
    "concat", lambda str_list: "-".join(str(s) for s in str_list)
)
OmegaConf.register_new_resolver(
    "multiply", lambda a, b: a * b
)
OmegaConf.register_new_resolver(
    "add", lambda a, b: a + b
)