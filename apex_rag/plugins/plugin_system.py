import logging
from typing import Any, Dict, List, Type

logger = logging.getLogger("apex_rag.plugins")


class BasePlugin:
    """Base class for all custom plugins in ApexRAG V3."""

    def __init__(self) -> None:
        self.name: str = self.__class__.__name__

    def initialize(self, index: Any) -> None:
        """Called during startup to pass reference to the parent ApexIndex instance."""
        pass


class PluginManager:
    """
    Evolved v3 plugin registry supporting custom Parsers, Verifiers,
    Agents, Graph Builders, Storage Backends, Retrieval Policies, and Synthesizers.
    """

    def __init__(self, index: Any) -> None:
        self.index = index
        self._plugins: Dict[str, BasePlugin] = {}
        self._custom_parsers: Dict[str, Type[Any]] = {}
        self._custom_verifiers: Dict[str, Type[Any]] = {}
        self._custom_agents: Dict[str, Type[Any]] = {}
        self._custom_graph_builders: Dict[str, Type[Any]] = {}
        self._custom_storage_backends: Dict[str, Type[Any]] = {}
        self._custom_retrieval_policies: Dict[str, Type[Any]] = {}
        self._custom_synthesizers: Dict[str, Type[Any]] = {}

    def register_plugin(self, plugin: BasePlugin) -> None:
        """Registers and initializes a plugin."""
        plugin.initialize(self.index)
        self._plugins[plugin.name] = plugin
        logger.info("Registered plugin: %s", plugin.name)

    def register_parser(self, extension: str, parser_cls: Type[Any]) -> None:
        self._custom_parsers[extension.lower()] = parser_cls

    def register_verifier(self, name: str, verifier_cls: Type[Any]) -> None:
        self._custom_verifiers[name.lower()] = verifier_cls

    def register_agent(self, role: str, agent_cls: Type[Any]) -> None:
        self._custom_agents[role.lower()] = agent_cls

    def register_graph_builder(self, name: str, builder_cls: Type[Any]) -> None:
        self._custom_graph_builders[name.lower()] = builder_cls

    def register_storage_backend(self, name: str, storage_cls: Type[Any]) -> None:
        self._custom_storage_backends[name.lower()] = storage_cls

    def register_retrieval_policy(self, name: str, policy_cls: Type[Any]) -> None:
        self._custom_retrieval_policies[name.lower()] = policy_cls

    def register_synthesizer(self, name: str, synth_cls: Type[Any]) -> None:
        self._custom_synthesizers[name.lower()] = synth_cls

    # -- Resolvers --
    def get_parser(self, extension: str) -> Optional[Type[Any]]:
        return self._custom_parsers.get(extension.lower())

    def get_verifier(self, name: str) -> Optional[Type[Any]]:
        return self._custom_verifiers.get(name.lower())

    def get_agent(self, role: str) -> Optional[Type[Any]]:
        return self._custom_agents.get(role.lower())

    def get_graph_builder(self, name: str) -> Optional[Type[Any]]:
        return self._custom_graph_builders.get(name.lower())

    def get_storage_backend(self, name: str) -> Optional[Type[Any]]:
        return self._custom_storage_backends.get(name.lower())

    def get_retrieval_policy(self, name: str) -> Optional[Type[Any]]:
        return self._custom_retrieval_policies.get(name.lower())

    def get_synthesizer(self, name: str) -> Optional[Type[Any]]:
        return self._custom_synthesizers.get(name.lower())
