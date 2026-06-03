from charmm_gui_md.shared.profile import SystemProfile
from charmm_gui_md.shared.protocols.schedule import ProtocolSchedule

SOLUTION_PROFILE = SystemProfile(
    initial_input_prefix="step3_input",
    protocol_schedule=ProtocolSchedule(
        stage_names=("step4_equilibration", "step5_production"),
    ),
)
