from charmm_gui_md.shared.profile import SystemProfile
from charmm_gui_md.shared.protocols.schedule import ProtocolSchedule

MEMBRANE_PROFILE = SystemProfile(
    initial_input_prefix="step5_input",
    protocol_schedule=ProtocolSchedule(
        stage_names=(
            "step6.1_equilibration",
            "step6.2_equilibration",
            "step6.3_equilibration",
            "step6.4_equilibration",
            "step6.5_equilibration",
            "step6.6_equilibration",
            "step7_production",
        ),
    ),
)
