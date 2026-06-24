import logging
from typing import Any

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
        self._plugins: dict[str, BasePlugin] = {}
        self._custom_parsers: dict[str, type[Any]] = {}
        self._custom_verifiers: dict[str, type[Any]] = {}
        self._custom_agents: dict[str, type[Any]] = {}
        self._custom_graph_builders: dict[str, type[Any]] = {}
        self._custom_storage_backends: dict[str, type[Any]] = {}
        self._custom_retrieval_policies: dict[str, type[Any]] = {}
        self._custom_synthesizers: dict[str, type[Any]] = {}

    def register_plugin(self, plugin: BasePlugin) -> None:
        """Registers and initializes a plugin."""
        plugin.initialize(self.index)
        self._plugins[plugin.name] = plugin
        logger.info("Registered plugin: %s", plugin.name)

    def register_parser(self, extension: str, parser_cls: type[Any]) -> None:
        self._custom_parsers[extension.lower()] = parser_cls

    def register_verifier(self, name: str, verifier_cls: type[Any]) -> None:
        self._custom_verifiers[name.lower()] = verifier_cls

    def register_agent(self, role: str, agent_cls: type[Any]) -> None:
        self._custom_agents[role.lower()] = agent_cls

    def register_graph_builder(self, name: str, builder_cls: type[Any]) -> None:
        self._custom_graph_builders[name.lower()] = builder_cls

    def register_storage_backend(self, name: str, storage_cls: type[Any]) -> None:
        self._custom_storage_backends[name.lower()] = storage_cls

    def register_retrieval_policy(self, name: str, policy_cls: type[Any]) -> None:
        self._custom_retrieval_policies[name.lower()] = policy_cls

    def register_synthesizer(self, name: str, synth_cls: type[Any]) -> None:
        self._custom_synthesizers[name.lower()] = synth_cls

    # -- Resolvers --
    def get_parser(self, extension: str) -> type[Any] | None:
        return self._custom_parsers.get(extension.lower())

    def get_verifier(self, name: str) -> type[Any] | None:
        return self._custom_verifiers.get(name.lower())

    def get_agent(self, role: str) -> type[Any] | None:
        return self._custom_agents.get(role.lower())

    def get_graph_builder(self, name: str) -> type[Any] | None:
        return self._custom_graph_builders.get(name.lower())

    def get_storage_backend(self, name: str) -> type[Any] | None:
        return self._custom_storage_backends.get(name.lower())

    def get_retrieval_policy(self, name: str) -> type[Any] | None:
        return self._custom_retrieval_policies.get(name.lower())

    def get_synthesizer(self, name: str) -> type[Any] | None:
        return self._custom_synthesizers.get(name.lower())
