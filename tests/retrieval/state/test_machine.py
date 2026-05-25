from apex_rag.retrieval.state.machine import RetrievalState, RetrievalStateMachine


def test_retrieval_state_machine_transitions():
    machine = RetrievalStateMachine(query_id="q-123")
    assert machine.current_state == RetrievalState.QUERY_RECEIVED
    assert len(machine.transitions) == 1

    machine.transition_to(RetrievalState.QUERY_CLASSIFIED, metadata={"type": "FACTUAL"})
    assert machine.current_state == RetrievalState.QUERY_CLASSIFIED
    assert len(machine.transitions) == 2

    trail = machine.get_audit_trail()
    assert len(trail) == 2
    assert trail[1]["from_state"] == "QUERY_RECEIVED"
    assert trail[1]["to_state"] == "QUERY_CLASSIFIED"
    assert trail[1]["metadata"]["type"] == "FACTUAL"
    assert trail[1]["duration_ms"] >= 0
