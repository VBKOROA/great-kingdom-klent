use great_kingdom_engine::game::ACTION_SPACE;

#[derive(Clone, Debug, PartialEq)]
pub struct GumbelResult {
    pub selected_action: Option<usize>,
    pub policy_target: [f32; ACTION_SPACE],
    pub visit_counts: [u32; ACTION_SPACE],
    pub root_value: f32,
}
