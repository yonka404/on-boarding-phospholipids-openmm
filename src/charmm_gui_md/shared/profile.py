from dataclasses import dataclass

from charmm_gui_md.shared.protocols.schedule import ProtocolSchedule


@dataclass(frozen=True)
class SystemProfile:
    initial_input_prefix: str
    protocol_schedule: ProtocolSchedule
