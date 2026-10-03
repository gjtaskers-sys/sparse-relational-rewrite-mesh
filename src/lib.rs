#![allow(dead_code)]

use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;

pub const NONE: u32 = u32::MAX;

pub type Label = u32;
pub type Relation = u32;
pub type RuleId = u32;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct NodeId(pub u32);

impl NodeId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct EdgeId(pub u32);

impl EdgeId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[inline]
fn clamp01(x: f32) -> f32 {
    if !x.is_finite() {
        0.0
    } else if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

#[inline]
fn clamp_nonneg(x: f32) -> f32 {
    if !x.is_finite() || x < 0.0 {
        0.0
    } else {
        x
    }
}

fn dedup_node_vec(v: &mut Vec<NodeId>) {
    v.sort_unstable();
    v.dedup();
}

fn dedup_edge_vec(v: &mut Vec<EdgeId>) {
    v.sort_unstable();
    v.dedup();
}

fn feature_key(feature: f32, quant: f32) -> u64 {
    if !feature.is_finite() {
        return 0;
    }

    if !quant.is_finite() || quant <= 0.0 {
        return feature.to_bits() as u64;
    }

    let mut scaled = (feature / quant).round();
    if scaled == 0.0 {
        scaled = 0.0;
    }

    if !scaled.is_finite() {
        0
    } else {
        scaled.to_bits() as u64
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub label: Label,
    pub feature: f32,
    pub activation: f32,
    pub bias: f32,
    pub alive: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub src: NodeId,
    pub dst: NodeId,
    pub relation: Relation,
    pub weight: f32,
    pub confidence: f32,
    pub next_out: u32,
    pub next_in: u32,
    pub alive: bool,
}

pub struct WorkingGraph {
    nodes: Vec<Node>,
    heads_out: Vec<u32>,
    heads_in: Vec<u32>,
    edges: Vec<Edge>,
    free_nodes: Vec<NodeId>,
    revision: u64,
}

impl WorkingGraph {
    pub fn new() -> Self {
        Self::with_capacity(0, 0)
    }

    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(nodes),
            heads_out: Vec::with_capacity(nodes),
            heads_in: Vec::with_capacity(nodes),
            edges: Vec::with_capacity(edges),
            free_nodes: Vec::new(),
            revision: 0,
        }
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    #[inline]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    #[inline]
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }

    #[inline]
    pub fn edge(&self, id: EdgeId) -> &Edge {
        &self.edges[id.index()]
    }

    #[inline]
    pub fn try_edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges.get(id.index())
    }

    #[inline]
    pub fn is_node_alive(&self, id: NodeId) -> bool {
        id.index() < self.nodes.len() && self.nodes[id.index()].alive
    }

    pub fn add_node(&mut self, label: Label, feature: f32, activation: f32) -> NodeId {
        self.allocate_node_placeholder(label, feature, activation)
    }

    pub fn try_add_edge(
        &mut self,
        src: NodeId,
        dst: NodeId,
        relation: Relation,
        weight: f32,
    ) -> Option<EdgeId> {
        if !self.is_node_alive(src) || !self.is_node_alive(dst) {
            return None;
        }

        let eid = self.allocate_edge_placeholder();
        self.init_edge(eid, src, dst, relation, weight);
        Some(eid)
    }

    pub fn reserve_nodes(&mut self, additional: usize) {
        self.nodes.reserve(additional);
        self.heads_out.reserve(additional);
        self.heads_in.reserve(additional);
    }

    pub fn reserve_edges(&mut self, additional: usize) {
        self.edges.reserve(additional);
    }

    fn allocate_node_placeholder(
        &mut self,
        label: Label,
        feature: f32,
        activation: f32,
    ) -> NodeId {
        let activation = clamp_nonneg(activation);

        let node = Node {
            label,
            feature,
            activation,
            bias: 0.0,
            alive: true,
        };

        if let Some(id) = self.free_nodes.pop() {
            let idx = id.index();
            self.nodes[idx] = node;
            self.heads_out[idx] = NONE;
            self.heads_in[idx] = NONE;
            id
        } else {
            assert!(
                self.nodes.len() < u32::MAX as usize,
                "node capacity overflow"
            );
            let id = NodeId(self.nodes.len() as u32);
            self.nodes.push(node);
            self.heads_out.push(NONE);
            self.heads_in.push(NONE);
            id
        }
    }

    fn allocate_edge_placeholder(&mut self) -> EdgeId {
        assert!(
            self.edges.len() < u32::MAX as usize,
            "edge capacity overflow"
        );

        let eid = EdgeId(self.edges.len() as u32);
        self.edges.push(Edge {
            src: NodeId(0),
            dst: NodeId(0),
            relation: 0,
            weight: 0.0,
            confidence: 0.0,
            next_out: NONE,
            next_in: NONE,
            alive: false,
        });
        eid
    }

    fn init_edge(
        &mut self,
        eid: EdgeId,
        src: NodeId,
        dst: NodeId,
        relation: Relation,
        weight: f32,
    ) {
        let idx = eid.index();
        let w = clamp_nonneg(weight);
        let next_out = self.heads_out[src.index()];
        let next_in = self.heads_in[dst.index()];

        self.edges[idx] = Edge {
            src,
            dst,
            relation,
            weight: w,
            confidence: clamp01(w),
            next_out,
            next_in,
            alive: true,
        };

        self.heads_out[src.index()] = eid.0;
        self.heads_in[dst.index()] = eid.0;
    }

    pub fn find_edge(&self, src: NodeId, dst: NodeId, relation: Relation) -> Option<EdgeId> {
        for eid in self.out_edges(src) {
            let e = &self.edges[eid.index()];
            if e.dst == dst && e.relation == relation {
                return Some(eid);
            }
        }
        None
    }

    pub fn find_best_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        relation: Option<Relation>,
        min_weight: f32,
    ) -> Option<(EdgeId, f32)> {
        let min_w = if min_weight.is_finite() { min_weight } else { 0.0 };
        let mut best: Option<(EdgeId, f32)> = None;

        for eid in self.out_edges(src) {
            let e = &self.edges[eid.index()];

            if e.dst != dst {
                continue;
            }

            if let Some(rel) = relation {
                if e.relation != rel {
                    continue;
                }
            }

            if !e.weight.is_finite() || e.weight < min_w {
                continue;
            }

            let better = match best {
                None => true,
                Some((_, old_w)) => e.weight > old_w,
            };

            if better {
                best = Some((eid, e.weight));
            }
        }

        best
    }

    pub fn out_edges(&self, node: NodeId) -> OutEdges<'_> {
        let cur = if node.index() < self.heads_out.len() {
            self.heads_out[node.index()]
        } else {
            NONE
        };

        OutEdges { graph: self, cur }
    }

    pub fn in_edges(&self, node: NodeId) -> InEdges<'_> {
        let cur = if node.index() < self.heads_in.len() {
            self.heads_in[node.index()]
        } else {
            NONE
        };

        InEdges { graph: self, cur }
    }

    pub fn consolidate(&mut self, params: &ConsolidationParams) {
        let threshold = if params.prune_weight_threshold.is_finite()
            && params.prune_weight_threshold > 0.0
        {
            params.prune_weight_threshold
        } else {
            0.0
        };

        let quant = if params.feature_quantization.is_finite()
            && params.feature_quantization > 0.0
        {
            params.feature_quantization
        } else {
            0.0
        };

        // 1. Prune invalid / weak edges.
        {
            let nodes = &self.nodes;
            for edge in self.edges.iter_mut() {
                if !edge.alive {
                    continue;
                }

                let src_ok = edge.src.index() < nodes.len() && nodes[edge.src.index()].alive;
                let dst_ok = edge.dst.index() < nodes.len() && nodes[edge.dst.index()].alive;

                if !src_ok || !dst_ok || !edge.weight.is_finite() || edge.weight.abs() < threshold {
                    edge.alive = false;
                }
            }
        }

        // 2. Merge duplicate nodes by label + quantized feature.
        let mut redirect: Vec<NodeId> = (0..self.nodes.len())
            .map(|i| NodeId(i as u32))
            .collect();

        let mut seen: HashMap<(Label, u64), NodeId> = HashMap::new();

        for i in 0..self.nodes.len() {
            if !self.nodes[i].alive {
                continue;
            }

            let key = (self.nodes[i].label, feature_key(self.nodes[i].feature, quant));

            if let Some(&canon) = seen.get(&key) {
                if canon.index() != i {
                    let dup_act = self.nodes[i].activation;
                    let dup_bias = self.nodes[i].bias;

                    self.nodes[i].alive = false;

                    let c = canon.index();
                    self.nodes[c].activation = self.nodes[c].activation.max(dup_act);
                    self.nodes[c].bias += dup_bias;

                    redirect[i] = canon;
                }
            } else {
                seen.insert(key, NodeId(i as u32));
            }
        }

        // 3. Redirect edges away from merged nodes.
        for edge in self.edges.iter_mut() {
            if !edge.alive {
                continue;
            }

            let src = redirect[edge.src.index()];
            let dst = redirect[edge.dst.index()];

            edge.src = src;
            edge.dst = dst;

            if src == dst {
                edge.alive = false;
            }
        }

        // 4. Rebuild compact adjacency lists.
        let old_edges = std::mem::take(&mut self.edges);

        for h in self.heads_out.iter_mut() {
            *h = NONE;
        }
        for h in self.heads_in.iter_mut() {
            *h = NONE;
        }

        let mut new_edges: Vec<Edge> = Vec::with_capacity(old_edges.len());

        let mut merge_map: HashMap<(u32, u32, Relation), usize> =
            if params.merge_parallel_edges {
                HashMap::with_capacity(old_edges.len())
            } else {
                HashMap::new()
            };

        for old in old_edges.iter() {
            if !old.alive {
                continue;
            }

            let src = old.src;
            let dst = old.dst;

            if src == dst {
                continue;
            }

            if src.index() >= self.nodes.len() || dst.index() >= self.nodes.len() {
                continue;
            }

            if !self.nodes[src.index()].alive || !self.nodes[dst.index()].alive {
                continue;
            }

            if !old.weight.is_finite() || old.weight.abs() < threshold {
                continue;
            }

            if params.merge_parallel_edges {
                let key = (src.0, dst.0, old.relation);

                if let Some(&idx) = merge_map.get(&key) {
                    let e = &mut new_edges[idx];
                    e.weight = e.weight.max(old.weight);
                    e.confidence = e.confidence.max(old.confidence);
                    continue;
                }

                let idx = new_edges.len();
                merge_map.insert(key, idx);

                let eid = idx as u32;
                new_edges.push(Edge {
                    src,
                    dst,
                    relation: old.relation,
                    weight: old.weight,
                    confidence: old.confidence,
                    next_out: self.heads_out[src.index()],
                    next_in: self.heads_in[dst.index()],
                    alive: true,
                });

                self.heads_out[src.index()] = eid;
                self.heads_in[dst.index()] = eid;
            } else {
                let eid = new_edges.len() as u32;
                new_edges.push(Edge {
                    src,
                    dst,
                    relation: old.relation,
                    weight: old.weight,
                    confidence: old.confidence,
                    next_out: self.heads_out[src.index()],
                    next_in: self.heads_in[dst.index()],
                    alive: true,
                });

                self.heads_out[src.index()] = eid;
                self.heads_in[dst.index()] = eid;
            }
        }

        self.edges = new_edges;

        self.free_nodes.clear();
        for i in 0..self.nodes.len() {
            if !self.nodes[i].alive {
                self.free_nodes.push(NodeId(i as u32));
            }
        }

        self.revision = self.revision.wrapping_add(1);
    }

    fn raw(&mut self) -> RawGraph {
        RawGraph {
            nodes: self.nodes.as_mut_ptr(),
            heads_out: self.heads_out.as_mut_ptr(),
            heads_in: self.heads_in.as_mut_ptr(),
            edges: self.edges.as_mut_ptr(),
            node_len: self.nodes.len(),
            edge_len: self.edges.len(),
        }
    }
}

pub struct OutEdges<'a> {
    graph: &'a WorkingGraph,
    cur: u32,
}

impl<'a> Iterator for OutEdges<'a> {
    type Item = EdgeId;

    fn next(&mut self) -> Option<EdgeId> {
        while self.cur != NONE {
            if self.cur as usize >= self.graph.edges.len() {
                self.cur = NONE;
                return None;
            }

            let eid = EdgeId(self.cur);
            let edge = &self.graph.edges[eid.index()];
            self.cur = edge.next_out;

            if edge.alive {
                return Some(eid);
            }
        }

        None
    }
}

pub struct InEdges<'a> {
    graph: &'a WorkingGraph,
    cur: u32,
}

impl<'a> Iterator for InEdges<'a> {
    type Item = EdgeId;

    fn next(&mut self) -> Option<EdgeId> {
        while self.cur != NONE {
            if self.cur as usize >= self.graph.edges.len() {
                self.cur = NONE;
                return None;
            }

            let eid = EdgeId(self.cur);
            let edge = &self.graph.edges[eid.index()];
            self.cur = edge.next_in;

            if edge.alive {
                return Some(eid);
            }
        }

        None
    }
}

#[derive(Clone, Copy)]
struct RawGraph {
    nodes: *mut Node,
    heads_out: *mut u32,
    heads_in: *mut u32,
    edges: *mut Edge,
    node_len: usize,
    edge_len: usize,
}

unsafe impl Send for RawGraph {}
unsafe impl Sync for RawGraph {}

impl RawGraph {
    #[inline]
    unsafe fn init_edge(
        self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        relation: Relation,
        weight: f32,
    ) {
        if id.index() >= self.edge_len || src.index() >= self.node_len || dst.index() >= self.node_len {
            return;
        }

        let src_node = self.nodes.add(src.index());
        let dst_node = self.nodes.add(dst.index());

        if !(*src_node).alive || !(*dst_node).alive {
            return;
        }

        let e = self.edges.add(id.index());

        let next_out = *self.heads_out.add(src.index());
        let next_in = *self.heads_in.add(dst.index());
        let w = clamp_nonneg(weight);

        *e = Edge {
            src,
            dst,
            relation,
            weight: w,
            confidence: clamp01(w),
            next_out,
            next_in,
            alive: true,
        };

        *self.heads_out.add(src.index()) = id.0;
        *self.heads_in.add(dst.index()) = id.0;
    }

    #[inline]
    unsafe fn reinforce_edge(self, id: EdgeId, amount: f32, inh: &InhibitionParams) {
        if id.index() >= self.edge_len {
            return;
        }

        let amount = if amount.is_finite() && amount > 0.0 {
            amount
        } else {
            return;
        };

        let e = self.edges.add(id.index());
        if !(*e).alive {
            return;
        }

        let src = (*e).src;
        if src.index() >= self.node_len {
            return;
        }

        let relation = (*e).relation;

        let old_weight = if (*e).weight.is_finite() {
            (*e).weight
        } else {
            0.0
        };

        let mut new_weight = old_weight + amount;
        if !new_weight.is_finite() || new_weight < 0.0 {
            new_weight = 0.0;
        }

        if inh.max_weight.is_finite() && new_weight > inh.max_weight {
            new_weight = inh.max_weight;
        }

        (*e).weight = new_weight;
        (*e).confidence = clamp01((*e).confidence + amount * 0.1);

        if !inh.enable {
            return;
        }

        let factor = if inh.decay_factor.is_finite() {
            inh.decay_factor.clamp(0.0, 1.0)
        } else {
            1.0
        };

        if factor >= 1.0 {
            return;
        }

        let mut cur = *self.heads_out.add(src.index());

        while cur != NONE {
            if cur as usize >= self.edge_len {
                break;
            }

            let ce = self.edges.add(cur as usize);

            if cur != id.0 && (*ce).alive {
                let should_decay = !inh.same_relation_only || (*ce).relation == relation;

                if should_decay {
                    let w = if (*ce).weight.is_finite() {
                        (*ce).weight
                    } else {
                        0.0
                    };

                    (*ce).weight = clamp_nonneg(w * factor);
                    (*ce).confidence = clamp01((*ce).confidence * factor);
                }
            }

            cur = (*ce).next_out;
        }
    }

    #[inline]
    unsafe fn set_activation(self, id: NodeId, activation: f32) {
        if id.index() >= self.node_len {
            return;
        }

        let n = self.nodes.add(id.index());
        if !(*n).alive {
            return;
        }

        let mut a = activation;
        if !a.is_finite() || a < 0.0 {
            a = 0.0;
        }

        (*n).activation = a;
    }

    #[inline]
    unsafe fn delete_edge(self, id: EdgeId) {
        if id.index() >= self.edge_len {
            return;
        }

        let e = self.edges.add(id.index());
        (*e).alive = false;
        (*e).weight = 0.0;
        (*e).confidence = 0.0;
    }

    #[inline]
    unsafe fn delete_node(self, id: NodeId) {
        if id.index() >= self.node_len {
            return;
        }

        let n = self.nodes.add(id.index());
        (*n).alive = false;
        (*n).activation = 0.0;
    }
}

#[derive(Clone, Debug)]
pub struct ActivationParams {
    pub decay: f32,
    pub edge_gain: f32,
    pub active_threshold: f32,
    pub max_activation: f32,
    pub max_steps: u32,
    pub convergence_epsilon: f32,
}

impl Default for ActivationParams {
    fn default() -> Self {
        Self {
            decay: 0.85,
            edge_gain: 0.10,
            active_threshold: 0.05,
            max_activation: 10.0,
            max_steps: 6,
            convergence_epsilon: 1e-4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MatcherParams {
    pub max_radius: u32,
    pub max_expansions: u32,
    pub max_candidates_per_node: usize,
    pub max_reachable_nodes: usize,
    pub max_matches_per_rule: usize,
}

impl Default for MatcherParams {
    fn default() -> Self {
        Self {
            max_radius: 2,
            max_expansions: 4096,
            max_candidates_per_node: 16,
            max_reachable_nodes: 4096,
            max_matches_per_rule: 64,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EnergyParams {
    pub max_energy: f32,
    pub min_confidence: f32,
    pub max_rewrites_per_cycle: usize,
}

impl Default for EnergyParams {
    fn default() -> Self {
        Self {
            max_energy: 0.0,
            min_confidence: 0.05,
            max_rewrites_per_cycle: 256,
        }
    }
}

#[derive(Clone, Debug)]
pub struct InhibitionParams {
    pub enable: bool,
    pub decay_factor: f32,
    pub same_relation_only: bool,
    pub max_weight: f32,
}

impl Default for InhibitionParams {
    fn default() -> Self {
        Self {
            enable: true,
            decay_factor: 0.98,
            same_relation_only: false,
            max_weight: 100.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConsolidationParams {
    pub prune_weight_threshold: f32,
    pub feature_quantization: f32,
    pub merge_parallel_edges: bool,
    pub inhibition: InhibitionParams,
}

impl Default for ConsolidationParams {
    fn default() -> Self {
        Self {
            prune_weight_threshold: 1e-3,
            feature_quantization: 0.01,
            merge_parallel_edges: true,
            inhibition: InhibitionParams::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EngineParams {
    pub activation: ActivationParams,
    pub matcher: MatcherParams,
    pub energy: EnergyParams,
    pub consolidation: ConsolidationParams,
}

impl Default for EngineParams {
    fn default() -> Self {
        Self {
            activation: ActivationParams::default(),
            matcher: MatcherParams::default(),
            energy: EnergyParams::default(),
            consolidation: ConsolidationParams::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PatternNode {
    pub label: Option<Label>,
    pub feature: Option<f32>,
    pub tolerance: f32,
    pub min_activation: f32,
}

#[derive(Clone, Debug)]
pub struct PatternEdge {
    pub src: u32,
    pub dst: u32,
    pub relation: Option<Relation>,
    pub min_weight: f32,
}

#[derive(Clone, Debug)]
pub struct Pattern {
    pub nodes: Vec<PatternNode>,
    pub edges: Vec<PatternEdge>,
}

#[derive(Clone, Copy, Debug)]
pub enum Endpoint {
    Matched(u32),
    New(u32),
}

#[derive(Clone, Debug)]
pub struct NewNodeSpec {
    pub label: Label,
    pub feature: f32,
    pub activation: f32,
}

#[derive(Clone, Debug)]
pub struct AddEdgeSpec {
    pub src: Endpoint,
    pub dst: Endpoint,
    pub relation: Relation,
    pub weight: f32,
}

#[derive(Clone, Debug)]
pub struct ActivationUpdateSpec {
    pub pattern_node: u32,
    pub activation: f32,
}

#[derive(Clone, Debug)]
pub struct Rule {
    pub id: RuleId,
    pub pattern: Pattern,
    pub cost: f32,
    pub gain: f32,
    pub max_radius: Option<u32>,
    pub min_confidence: f32,
    pub reinforce_matched: f32,

    pub delete_pattern_edges: Vec<u32>,
    pub delete_pattern_nodes: Vec<u32>,

    pub add_nodes: Vec<NewNodeSpec>,
    pub add_edges: Vec<AddEdgeSpec>,

    pub activation_updates: Vec<ActivationUpdateSpec>,
}

pub struct CompiledRule {
    pub index: usize,
    pub rule: Rule,
    pub order: Vec<u32>,
    pub incident: Vec<Vec<u32>>,
}

impl CompiledRule {
    fn compile(rule: Rule, index: usize) -> Self {
        let n = rule.pattern.nodes.len();
        assert!(n > 0, "rule pattern must contain at least one node");

        for e in &rule.pattern.edges {
            assert!(
                (e.src as usize) < n && (e.dst as usize) < n,
                "pattern edge endpoints must be valid"
            );
        }

        for idx in &rule.delete_pattern_edges {
            assert!(
                (*idx as usize) < rule.pattern.edges.len(),
                "delete_pattern_edges index out of range"
            );
        }

        for idx in &rule.delete_pattern_nodes {
            assert!(
                (*idx as usize) < n,
                "delete_pattern_nodes index out of range"
            );
        }

        for up in &rule.activation_updates {
            assert!(
                (up.pattern_node as usize) < n,
                "activation_updates index out of range"
            );
        }

        for add in &rule.add_edges {
            match add.src {
                Endpoint::Matched(i) => assert!((i as usize) < n, "add_edge src pattern index out of range"),
                Endpoint::New(i) => assert!((i as usize) < rule.add_nodes.len(), "add_edge src new-node index out of range"),
            }

            match add.dst {
                Endpoint::Matched(i) => assert!((i as usize) < n, "add_edge dst pattern index out of range"),
                Endpoint::New(i) => assert!((i as usize) < rule.add_nodes.len(), "add_edge dst new-node index out of range"),
            }
        }

        let mut incident = vec![Vec::new(); n];

        for (ei, e) in rule.pattern.edges.iter().enumerate() {
            let s = e.src as usize;
            let d = e.dst as usize;

            incident[s].push(ei as u32);
            if d != s {
                incident[d].push(ei as u32);
            }
        }

        let mut order: Vec<u32> = (1..n as u32).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(incident[i as usize].len()));
        order.insert(0, 0);

        Self {
            index,
            rule,
            order,
            incident,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MatchCandidate {
    pub rule_index: usize,
    pub energy: f32,
    pub confidence: f32,
    pub nodes: Vec<NodeId>,
    pub reinforce_edges: Vec<EdgeId>,
    pub delete_edges: Vec<EdgeId>,
    pub delete_nodes: Vec<NodeId>,
    pub activation_updates: Vec<(NodeId, f32)>,
    pub touched_nodes: Vec<NodeId>,
    pub touched_edges: Vec<EdgeId>,
}

fn node_compatible(node: &Node, pn: &PatternNode) -> bool {
    if !node.alive {
        return false;
    }

    let min_act = if pn.min_activation.is_finite() {
        pn.min_activation
    } else {
        0.0
    };

    if node.activation < min_act {
        return false;
    }

    if let Some(label) = pn.label {
        if node.label != label {
            return false;
        }
    }

    if let Some(f) = pn.feature {
        if !f.is_finite() || !node.feature.is_finite() {
            return false;
        }

        let tol = if pn.tolerance.is_finite() && pn.tolerance >= 0.0 {
            pn.tolerance
        } else {
            0.0
        };

        if (node.feature - f).abs() > tol {
            return false;
        }
    }

    true
}

struct RuleMatcher<'a> {
    graph: &'a WorkingGraph,
    compiled: &'a CompiledRule,
    params: &'a EngineParams,

    visited: Vec<u32>,
    dist: Vec<u32>,
    stamp: u32,

    queue_nodes: Vec<NodeId>,
    queue_depth: Vec<u32>,
    reachable: Vec<NodeId>,

    candidates: Vec<Vec<NodeId>>,

    assign_nodes: Vec<Option<NodeId>>,
    assign_edges: Vec<Option<EdgeId>>,

    used: Vec<u32>,
    used_stamp: u32,

    edge_undo: Vec<(usize, Option<EdgeId>)>,
}

impl<'a> RuleMatcher<'a> {
    fn new(
        graph: &'a WorkingGraph,
        compiled: &'a CompiledRule,
        params: &'a EngineParams,
    ) -> Self {
        let n = graph.node_count();
        let pn = compiled.rule.pattern.nodes.len();
        let pe = compiled.rule.pattern.edges.len();

        Self {
            graph,
            compiled,
            params,
            visited: vec![0; n],
            dist: vec![0; n],
            stamp: 0,
            queue_nodes: Vec::new(),
            queue_depth: Vec::new(),
            reachable: Vec::new(),
            candidates: vec![Vec::new(); pn],
            assign_nodes: vec![None; pn],
            assign_edges: vec![None; pe],
            used: vec![0; n],
            used_stamp: 0,
            edge_undo: Vec::new(),
        }
    }

    fn next_visit_stamp(&mut self) -> u32 {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.visited.fill(0);
            self.stamp = 1;
        }
        self.stamp
    }

    fn next_used_stamp(&mut self) -> u32 {
        self.used_stamp = self.used_stamp.wrapping_add(1);
        if self.used_stamp == 0 {
            self.used.fill(0);
            self.used_stamp = 1;
        }
        self.used_stamp
    }

    fn try_seed(&mut self, seed: NodeId) -> Option<MatchCandidate> {
        if seed.index() >= self.graph.node_count() {
            return None;
        }

        let pattern = &self.compiled.rule.pattern;

        if pattern.nodes.is_empty() {
            return None;
        }

        if !self.graph.is_node_alive(seed) {
            return None;
        }

        if !node_compatible(self.graph.node(seed), &pattern.nodes[0]) {
            return None;
        }

        let global_radius = self.params.matcher.max_radius;
        let rule_radius = self.compiled.rule.max_radius.unwrap_or(global_radius);
        let radius = rule_radius.min(global_radius);

        let max_reachable = self.params.matcher.max_reachable_nodes.max(1);
        let max_candidates = self.params.matcher.max_candidates_per_node.max(1);

        let stamp = self.next_visit_stamp();

        self.queue_nodes.clear();
        self.queue_depth.clear();
        self.reachable.clear();

        self.visited[seed.index()] = stamp;
        self.dist[seed.index()] = 0;

        self.queue_nodes.push(seed);
        self.queue_depth.push(0);
        self.reachable.push(seed);

        let mut head = 0usize;

        'bfs: while head < self.queue_nodes.len() {
            if self.reachable.len() >= max_reachable {
                break;
            }

            let node = self.queue_nodes[head];
            let depth = self.queue_depth[head];
            head += 1;

            if depth >= radius {
                continue;
            }

            for eid in self.graph.out_edges(node) {
                let e = self.graph.edge(eid);
                let nb = e.dst;

                if nb.index() < self.visited.len()
                    && self.graph.is_node_alive(nb)
                    && self.visited[nb.index()] != stamp
                {
                    self.visited[nb.index()] = stamp;
                    self.dist[nb.index()] = depth + 1;

                    self.queue_nodes.push(nb);
                    self.queue_depth.push(depth + 1);
                    self.reachable.push(nb);

                    if self.reachable.len() >= max_reachable {
                        break 'bfs;
                    }
                }
            }

            for eid in self.graph.in_edges(node) {
                let e = self.graph.edge(eid);
                let nb = e.src;

                if nb.index() < self.visited.len()
                    && self.graph.is_node_alive(nb)
                    && self.visited[nb.index()] != stamp
                {
                    self.visited[nb.index()] = stamp;
                    self.dist[nb.index()] = depth + 1;

                    self.queue_nodes.push(nb);
                    self.queue_depth.push(depth + 1);
                    self.reachable.push(nb);

                    if self.reachable.len() >= max_reachable {
                        break 'bfs;
                    }
                }
            }
        }

        for c in self.candidates.iter_mut() {
            c.clear();
        }

        for p in 0..pattern.nodes.len() {
            if p == 0 {
                self.candidates[0].push(seed);
            } else {
                let pn = &pattern.nodes[p];

                for &nid in &self.reachable {
                    if nid.index() < self.graph.node_count()
                        && node_compatible(self.graph.node(nid), pn)
                    {
                        self.candidates[p].push(nid);
                    }
                }

                if self.candidates[p].len() > max_candidates {
                    let dist = &self.dist;
                    let graph = self.graph;
                    let cand = &mut self.candidates[p];

                    cand.sort_unstable_by(|a, b| {
                        let da = dist[a.index()];
                        let db = dist[b.index()];

                        da.cmp(&db).then_with(|| {
                            let aa = graph.node(*a).activation;
                            let ab = graph.node(*b).activation;
                            ab.partial_cmp(&aa).unwrap_or(Ordering::Equal)
                        })
                    });

                    cand.truncate(max_candidates);
                }
            }

            if self.candidates[p].is_empty() {
                return None;
            }
        }

        let used_stamp = self.next_used_stamp();

        self.assign_nodes.fill(None);
        self.assign_edges.fill(None);
        self.edge_undo.clear();

        self.assign_nodes[0] = Some(seed);
        self.used[seed.index()] = used_stamp;

        let mut expansions = 0u32;
        let ok = self.search(1, &mut expansions, radius);

        if !ok {
            return None;
        }

        self.edge_undo.clear();
        self.finalize_candidate()
    }

    fn search(&mut self, pos: usize, expansions: &mut u32, radius: u32) -> bool {
        let order_len = self.compiled.order.len();

        if pos >= order_len {
            return true;
        }

        let max_exp = self.params.matcher.max_expansions.max(1);
        let p = self.compiled.order[pos] as usize;
        let cand_len = self.candidates[p].len();

        for ci in 0..cand_len {
            if *expansions >= max_exp {
                return false;
            }

            *expansions = expansions.saturating_add(1);

            let h = self.candidates[p][ci];

            if h.index() >= self.dist.len() || self.dist[h.index()] > radius {
                continue;
            }

            if h.index() < self.used.len() && self.used[h.index()] == self.used_stamp {
                continue;
            }

            let undo_start = self.edge_undo.len();
            let mut ok = true;

            let incident_len = self.compiled.incident[p].len();

            for ii in 0..incident_len {
                let ei = self.compiled.incident[p][ii] as usize;

                let src_pat = self.compiled.rule.pattern.edges[ei].src as usize;
                let dst_pat = self.compiled.rule.pattern.edges[ei].dst as usize;
                let relation = self.compiled.rule.pattern.edges[ei].relation;
                let min_weight = self.compiled.rule.pattern.edges[ei].min_weight;

                let other = if src_pat == p { dst_pat } else { src_pat };

                let other_host = if other == p {
                    Some(h)
                } else {
                    self.assign_nodes[other]
                };

                if let Some(other_h) = other_host {
                    let (src_host, dst_host) = if src_pat == p {
                        (h, other_h)
                    } else if dst_pat == p {
                        (other_h, h)
                    } else {
                        continue;
                    };

                    match self.assign_edges[ei] {
                        Some(existing) => {
                            if let Some(e) = self.graph.try_edge(existing) {
                                if e.src != src_host || e.dst != dst_host {
                                    ok = false;
                                    break;
                                }
                            } else {
                                ok = false;
                                break;
                            }
                        }
                        None => {
                            if let Some((found, _)) = self.graph.find_best_edge(
                                src_host,
                                dst_host,
                                relation,
                                min_weight,
                            ) {
                                self.assign_edges[ei] = Some(found);
                                self.edge_undo.push((ei, None));
                            } else {
                                ok = false;
                                break;
                            }
                        }
                    }
                }
            }

            if !ok {
                self.undo_edges(undo_start);
                continue;
            }

            self.assign_nodes[p] = Some(h);
            if h.index() < self.used.len() {
                self.used[h.index()] = self.used_stamp;
            }

            if self.search(pos + 1, expansions, radius) {
                return true;
            }

            if h.index() < self.used.len() {
                self.used[h.index()] = 0;
            }
            self.assign_nodes[p] = None;
            self.undo_edges(undo_start);
        }

        false
    }

    fn undo_edges(&mut self, start: usize) {
        while self.edge_undo.len() > start {
            let (idx, prev) = self.edge_undo.pop().unwrap();
            self.assign_edges[idx] = prev;
        }
    }

    fn finalize_candidate(&mut self) -> Option<MatchCandidate> {
        let rule = &self.compiled.rule;
        let pattern = &rule.pattern;

        let mut nodes: Vec<NodeId> = Vec::with_capacity(pattern.nodes.len());

        for opt in &self.assign_nodes {
            match opt {
                Some(id) => nodes.push(*id),
                None => return None,
            }
        }

        let mut matched_edges: Vec<EdgeId> = Vec::with_capacity(pattern.edges.len());

        for (ei, pe) in pattern.edges.iter().enumerate() {
            let mut opt = self.assign_edges[ei];

            if opt.is_none() {
                let src = nodes[pe.src as usize];
                let dst = nodes[pe.dst as usize];

                match self.graph.find_best_edge(src, dst, pe.relation, pe.min_weight) {
                    Some((eid, _)) => {
                        opt = Some(eid);
                        self.assign_edges[ei] = Some(eid);
                    }
                    None => return None,
                }
            }

            matched_edges.push(opt.unwrap());
        }

        let mut mismatch = 0.0f32;

        for (i, pn) in pattern.nodes.iter().enumerate() {
            let host = self.graph.node(nodes[i]);

            if let Some(f) = pn.feature {
                if f.is_finite() && host.feature.is_finite() {
                    let tol = if pn.tolerance.is_finite() && pn.tolerance >= 0.0 {
                        pn.tolerance
                    } else {
                        0.0
                    };

                    let diff = (host.feature - f).abs();
                    mismatch += (diff - tol).max(0.0);
                }
            }

            let min_act = if pn.min_activation.is_finite() {
                pn.min_activation
            } else {
                0.0
            };

            mismatch += (min_act - host.activation).max(0.0);
        }

        for (ei, pe) in pattern.edges.iter().enumerate() {
            let e = self.graph.edge(matched_edges[ei]);
            mismatch += (pe.min_weight - e.weight).max(0.0);
        }

        if !mismatch.is_finite() {
            return None;
        }

        let max_act = self.params.activation.max_activation.max(1e-6);

        let act_sum: f32 = nodes
            .iter()
            .map(|id| self.graph.node(*id).activation)
            .sum();

        let node_conf = clamp01(act_sum / (nodes.len() as f32 * max_act));

        let edge_conf = if matched_edges.is_empty() {
            1.0
        } else {
            let w_sum: f32 = matched_edges
                .iter()
                .map(|eid| clamp01(self.graph.edge(*eid).weight))
                .sum();

            clamp01(w_sum / matched_edges.len() as f32)
        };

        let confidence = clamp01(node_conf * 0.5 + edge_conf * 0.5);

        let energy = mismatch + rule.cost - rule.gain * confidence;

        if !energy.is_finite() || !confidence.is_finite() {
            return None;
        }

        let rule_min = if rule.min_confidence.is_finite() {
            rule.min_confidence
        } else {
            0.0
        };

        let global_min = if self.params.energy.min_confidence.is_finite() {
            self.params.energy.min_confidence
        } else {
            0.0
        };

        let min_conf = rule_min.max(global_min).clamp(0.0, 1.0);

        if confidence < min_conf {
            return None;
        }

        let max_energy = if self.params.energy.max_energy.is_finite() {
            self.params.energy.max_energy
        } else {
            f32::INFINITY
        };

        if energy > max_energy {
            return None;
        }

        let mut delete_nodes: Vec<NodeId> = Vec::new();
        for idx in &rule.delete_pattern_nodes {
            if let Some(&nid) = nodes.get(*idx as usize) {
                delete_nodes.push(nid);
            }
        }
        dedup_node_vec(&mut delete_nodes);

        let mut delete_edges: Vec<EdgeId> = Vec::new();

        for idx in &rule.delete_pattern_edges {
            if let Some(&eid) = matched_edges.get(*idx as usize) {
                delete_edges.push(eid);
            }
        }

        for &nid in &delete_nodes {
            for eid in self.graph.out_edges(nid) {
                delete_edges.push(eid);
            }

            for eid in self.graph.in_edges(nid) {
                delete_edges.push(eid);
            }
        }

        dedup_edge_vec(&mut delete_edges);

        let reinforce_edges: Vec<EdgeId> = matched_edges
            .iter()
            .copied()
            .filter(|eid| delete_edges.binary_search(eid).is_err())
            .collect();

        let mut activation_updates: Vec<(NodeId, f32)> = Vec::new();

        for up in &rule.activation_updates {
            if let Some(&host) = nodes.get(up.pattern_node as usize) {
                if delete_nodes.binary_search(&host).is_err() {
                    activation_updates.push((host, up.activation));
                }
            }
        }

        let mut touched_nodes = nodes.clone();
        touched_nodes.extend(delete_nodes.iter().copied());

        let mut touched_edges = matched_edges.clone();
        touched_edges.extend(delete_edges.iter().copied());
        touched_edges.extend(reinforce_edges.iter().copied());

        for &eid in &touched_edges {
            if let Some(e) = self.graph.try_edge(eid) {
                touched_nodes.push(e.src);
                touched_nodes.push(e.dst);
            }
        }

        dedup_node_vec(&mut touched_nodes);
        dedup_edge_vec(&mut touched_edges);

        Some(MatchCandidate {
            rule_index: self.compiled.index,
            energy,
            confidence,
            nodes,
            reinforce_edges,
            delete_edges,
            delete_nodes,
            activation_updates,
            touched_nodes,
            touched_edges,
        })
    }
}

struct ConflictTracker {
    node_mark: Vec<u32>,
    edge_mark: Vec<u32>,
    stamp: u32,
}

impl ConflictTracker {
    fn new() -> Self {
        Self {
            node_mark: Vec::new(),
            edge_mark: Vec::new(),
            stamp: 0,
        }
    }

    fn ensure(&mut self, node_len: usize, edge_len: usize) {
        if self.node_mark.len() < node_len {
            self.node_mark.resize(node_len, 0);
        }

        if self.edge_mark.len() < edge_len {
            self.edge_mark.resize(edge_len, 0);
        }
    }

    fn next_stamp(&mut self) {
        self.stamp = self.stamp.wrapping_add(1);

        if self.stamp == 0 {
            self.node_mark.fill(0);
            self.edge_mark.fill(0);
            self.stamp = 1;
        }
    }

    fn try_reserve(&mut self, nodes: &[NodeId], edges: &[EdgeId]) -> bool {
        for n in nodes {
            if n.index() >= self.node_mark.len() || self.node_mark[n.index()] == self.stamp {
                return false;
            }
        }

        for e in edges {
            if e.index() >= self.edge_mark.len() || self.edge_mark[e.index()] == self.stamp {
                return false;
            }
        }

        for n in nodes {
            self.node_mark[n.index()] = self.stamp;
        }

        for e in edges {
            self.edge_mark[e.index()] = self.stamp;
        }

        true
    }
}

#[derive(Clone, Copy, Debug)]
struct NewEdgePlan {
    id: EdgeId,
    src: NodeId,
    dst: NodeId,
    relation: Relation,
    weight: f32,
}

#[derive(Clone, Debug)]
struct PlannedRewrite {
    new_edges: Vec<NewEdgePlan>,
    reinforce_edges: Vec<EdgeId>,
    delete_edges: Vec<EdgeId>,
    delete_nodes: Vec<NodeId>,
    activation_updates: Vec<(NodeId, f32)>,
    reinforce_amount: f32,
    inhibition: InhibitionParams,
}

pub struct Engine {
    graph: WorkingGraph,
    rules: Vec<CompiledRule>,
    params: EngineParams,
    conflict: ConflictTracker,
    activation_buffer: Vec<f32>,
    active_buffer: Vec<NodeId>,
}

impl Engine {
    pub fn new(params: EngineParams) -> Self {
        Self {
            graph: WorkingGraph::new(),
            rules: Vec::new(),
            params,
            conflict: ConflictTracker::new(),
            activation_buffer: Vec::new(),
            active_buffer: Vec::new(),
        }
    }

    pub fn graph(&self) -> &WorkingGraph {
        &self.graph
    }

    pub fn add_node(&mut self, label: Label, feature: f32, activation: f32) -> NodeId {
        self.graph.add_node(label, feature, activation)
    }

    pub fn try_add_edge(
        &mut self,
        src: NodeId,
        dst: NodeId,
        relation: Relation,
        weight: f32,
    ) -> Option<EdgeId> {
        self.graph.try_add_edge(src, dst, relation, weight)
    }

    pub fn add_rule(&mut self, mut rule: Rule) -> RuleId {
        let idx = self.rules.len();
        rule.id = idx as RuleId;

        let compiled = CompiledRule::compile(rule, idx);
        self.rules.push(compiled);

        idx as RuleId
    }

    pub fn run_cycles(&mut self, count: usize) -> usize {
        let mut total = 0;

        for _ in 0..count {
            total += self.run_cycle();
        }

        total
    }

    pub fn run_cycle(&mut self) -> usize {
        self.spread_activation();

        let active = std::mem::take(&mut self.active_buffer);

        if active.is_empty() {
            self.graph.consolidate(&self.params.consolidation);
            return 0;
        }

        let candidates = self.generate_candidates(&active);

        if candidates.is_empty() {
            self.graph.consolidate(&self.params.consolidation);
            return 0;
        }

        let selected = self.select_nonconflicting(candidates);
        let executed = selected.len();

        if !selected.is_empty() {
            let plans = self.prepare_plans(selected);
            Self::execute_plans_parallel(&mut self.graph, plans);
        }

        self.graph.consolidate(&self.params.consolidation);

        executed
    }

    fn spread_activation(&mut self) {
        let params = self.params.activation.clone();
        let n = self.graph.node_count();

        self.activation_buffer.resize(n, 0.0);
        self.active_buffer.clear();

        for i in 0..n {
            let node = &self.graph.nodes[i];

            if node.alive && node.activation >= params.active_threshold {
                self.active_buffer.push(NodeId(i as u32));
            }
        }

        if params.max_steps == 0 || self.active_buffer.is_empty() {
            return;
        }

        let decay = if params.decay.is_finite() { params.decay } else { 0.0 };
        let edge_gain = if params.edge_gain.is_finite() {
            params.edge_gain
        } else {
            0.0
        };

        let mut step = 0u32;
        let mut diff = f32::INFINITY;

        while step < params.max_steps
            && !self.active_buffer.is_empty()
            && diff > params.convergence_epsilon
        {
            self.activation_buffer.fill(0.0);

            let current = std::mem::take(&mut self.active_buffer);

            for &src in &current {
                if src.index() >= n {
                    continue;
                }

                let act = self.graph.nodes[src.index()].activation;

                if act <= 0.0 || !act.is_finite() {
                    continue;
                }

                for eid in self.graph.out_edges(src) {
                    let e = &self.graph.edges[eid.index()];
                    let dst = e.dst;

                    if dst.index() < n && self.graph.nodes[dst.index()].alive {
                        let contrib = act * e.weight * edge_gain;

                        if contrib.is_finite() {
                            self.activation_buffer[dst.index()] += contrib;
                        }
                    }
                }
            }

            diff = 0.0;
            self.active_buffer.clear();

            let incoming = &self.activation_buffer;

            for i in 0..n {
                let node = &mut self.graph.nodes[i];

                if !node.alive {
                    continue;
                }

                let old = node.activation;

                let mut new_act = old * decay + incoming[i] + node.bias;

                if !new_act.is_finite() || new_act < 0.0 {
                    new_act = 0.0;
                }

                if params.max_activation.is_finite() && new_act > params.max_activation {
                    new_act = params.max_activation;
                }

                node.activation = new_act;

                let delta = (new_act - old).abs();
                if delta.is_finite() && delta > diff {
                    diff = delta;
                }

                if new_act >= params.active_threshold {
                    self.active_buffer.push(NodeId(i as u32));
                }
            }

            step += 1;
        }
    }

    fn generate_candidates(&self, active: &[NodeId]) -> Vec<MatchCandidate> {
        let graph = &self.graph;
        let params = &self.params;
        let rules = &self.rules;

        let batches: Vec<Vec<MatchCandidate>> = rules
            .par_iter()
            .map(|compiled| {
                let mut out = Vec::new();

                if compiled.rule.pattern.nodes.is_empty() {
                    return out;
                }

                let mut matcher = RuleMatcher::new(graph, compiled, params);
                let max_matches = params.matcher.max_matches_per_rule;

                for &seed in active {
                    if out.len() >= max_matches {
                        break;
                    }

                    if let Some(candidate) = matcher.try_seed(seed) {
                        out.push(candidate);
                    }
                }

                out
            })
            .collect();

        batches.into_iter().flatten().collect()
    }

    fn select_nonconflicting(&mut self, mut candidates: Vec<MatchCandidate>) -> Vec<MatchCandidate> {
        candidates.sort_by(|a, b| {
            let energy_cmp = a
                .energy
                .partial_cmp(&b.energy)
                .unwrap_or(Ordering::Equal);

            let conf_cmp = b
                .confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(Ordering::Equal);

            energy_cmp.then(conf_cmp)
        });

        self.conflict
            .ensure(self.graph.node_count(), self.graph.edge_count());
        self.conflict.next_stamp();

        let mut selected = Vec::new();

        for c in candidates {
            if selected.len() >= self.params.energy.max_rewrites_per_cycle {
                break;
            }

            if self
                .conflict
                .try_reserve(&c.touched_nodes, &c.touched_edges)
            {
                selected.push(c);
            }
        }

        selected
    }

    fn prepare_plans(&mut self, selected: Vec<MatchCandidate>) -> Vec<PlannedRewrite> {
        let mut total_new_nodes = 0usize;
        let mut total_new_edges = 0usize;

        for c in &selected {
            let rule = &self.rules[c.rule_index].rule;
            total_new_nodes += rule.add_nodes.len();
            total_new_edges += rule.add_edges.len();
        }

        self.graph.reserve_nodes(total_new_nodes);
        self.graph.reserve_edges(total_new_edges);

        let mut plans = Vec::with_capacity(selected.len());

        for c in selected {
            let rule_index = c.rule_index;
            let rule = &self.rules[rule_index].rule;

            let mut new_node_ids = Vec::with_capacity(rule.add_nodes.len());

            for spec in &rule.add_nodes {
                let id = self
                    .graph
                    .allocate_node_placeholder(spec.label, spec.feature, spec.activation);
                new_node_ids.push(id);
            }

            let mut plan = PlannedRewrite {
                new_edges: Vec::with_capacity(rule.add_edges.len()),
                reinforce_edges: c.reinforce_edges,
                delete_edges: c.delete_edges,
                delete_nodes: c.delete_nodes,
                activation_updates: c.activation_updates,
                reinforce_amount: rule.reinforce_matched,
                inhibition: self.params.consolidation.inhibition.clone(),
            };

            for spec in &rule.add_edges {
                let src = resolve_endpoint(&spec.src, &c.nodes, &new_node_ids);
                let dst = resolve_endpoint(&spec.dst, &c.nodes, &new_node_ids);

                if let (Some(src), Some(dst)) = (src, dst) {
                    if self.graph.is_node_alive(src) && self.graph.is_node_alive(dst) {
                        if let Some(existing) = self.graph.find_edge(src, dst, spec.relation) {
                            if plan.delete_edges.binary_search(&existing).is_err() {
                                plan.reinforce_edges.push(existing);
                            }
                        } else {
                            let eid = self.graph.allocate_edge_placeholder();

                            plan.new_edges.push(NewEdgePlan {
                                id: eid,
                                src,
                                dst,
                                relation: spec.relation,
                                weight: spec.weight,
                            });
                        }
                    }
                }
            }

            dedup_edge_vec(&mut plan.reinforce_edges);

            if !plan.delete_edges.is_empty() {
                plan.reinforce_edges
                    .retain(|eid| plan.delete_edges.binary_search(eid).is_err());
            }

            plans.push(plan);
        }

        plans
    }

    fn execute_plans_parallel(graph: &mut WorkingGraph, plans: Vec<PlannedRewrite>) {
        if plans.is_empty() {
            return;
        }

        let raw = graph.raw();

        plans
            .par_iter()
            .for_each(|plan| unsafe { apply_plan(raw, plan) });
    }
}

fn resolve_endpoint(
    ep: &Endpoint,
    matched: &[NodeId],
    new_nodes: &[NodeId],
) -> Option<NodeId> {
    match ep {
        Endpoint::Matched(i) => matched.get(*i as usize).copied(),
        Endpoint::New(i) => new_nodes.get(*i as usize).copied(),
    }
}

unsafe fn apply_plan(raw: RawGraph, plan: &PlannedRewrite) {
    for ne in &plan.new_edges {
        raw.init_edge(ne.id, ne.src, ne.dst, ne.relation, ne.weight);
    }

    if plan.reinforce_amount.is_finite() && plan.reinforce_amount > 0.0 {
        for &eid in &plan.reinforce_edges {
            raw.reinforce_edge(eid, plan.reinforce_amount, &plan.inhibition);
        }
    }

    for &(nid, act) in &plan.activation_updates {
        raw.set_activation(nid, act);
    }

    for &eid in &plan.delete_edges {
        raw.delete_edge(eid);
    }

    for &nid in &plan.delete_nodes {
        raw.delete_node(nid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reinforcement_and_competitive_inhibition() {
        let mut params = EngineParams::default();

        params.activation.decay = 1.0;
        params.activation.edge_gain = 0.0;
        params.activation.max_steps = 1;
        params.activation.active_threshold = 0.01;

        params.matcher.max_radius = 1;
        params.matcher.max_expansions = 1000;

        params.energy.max_energy = 10.0;
        params.energy.min_confidence = 0.0;

        params.consolidation.prune_weight_threshold = 0.001;
        params.consolidation.inhibition.enable = true;
        params.consolidation.inhibition.decay_factor = 0.5;
        params.consolidation.inhibition.same_relation_only = false;

        let mut engine = Engine::new(params);

        let a = engine.add_node(1, 0.0, 1.0);
        let b = engine.add_node(2, 0.0, 1.0);
        let c = engine.add_node(3, 0.0, 1.0);

        engine.try_add_edge(a, b, 1, 0.8).unwrap();
        engine.try_add_edge(a, c, 2, 0.6).unwrap();

        let rule = Rule {
            id: 0,
            pattern: Pattern {
                nodes: vec![
                    PatternNode {
                        label: Some(1),
                        feature: None,
                        tolerance: 0.1,
                        min_activation: 0.0,
                    },
                    PatternNode {
                        label: Some(2),
                        feature: None,
                        tolerance: 0.1,
                        min_activation: 0.0,
                    },
                ],
                edges: vec![PatternEdge {
                    src: 0,
                    dst: 1,
                    relation: Some(1),
                    min_weight: 0.5,
                }],
            },
            cost: 0.0,
            gain: 1.0,
            max_radius: Some(1),
            min_confidence: 0.0,
            reinforce_matched: 0.2,
            delete_pattern_edges: vec![],
            delete_pattern_nodes: vec![],
            add_nodes: vec![],
            add_edges: vec![AddEdgeSpec {
                src: Endpoint::Matched(0),
                dst: Endpoint::Matched(1),
                relation: 3,
                weight: 0.9,
            }],
            activation_updates: vec![],
        };

        engine.add_rule(rule);

        let executed = engine.run_cycle();
        assert_eq!(executed, 1);

        let g = engine.graph();

        let ab = g.find_edge(a, b, 1).expect("edge A->B relation 1 should exist");
        let ac = g.find_edge(a, c, 2).expect("edge A->C relation 2 should exist");
        let ab_new = g.find_edge(a, b, 3).expect("new edge A->B relation 3 should exist");

        assert!(g.edge(ab).weight >= 0.99);
        assert!(g.edge(ac).weight <= 0.31);
        assert!(g.edge(ab_new).weight > 0.0);
    }
}