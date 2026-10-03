use sparse_relational_rewrite_mesh::{Node, NodeId, Edge};
use std::io::{self, Write};

fn main() {
    println!("====================================================");
    println!("🧠 SPARSE RELATIONAL REWRITE MESH INTERACTIVE CONSOLE");
    println!("====================================================");

    // 1. Initialize our memory pools
    let mut node_arena: Vec<Node> = vec![
        Node { label: 101, feature: 0.8, activation: 0.0, bias: 0.0, alive: true }, // Concept A
        Node { label: 102, feature: 0.5, activation: 0.0, bias: 0.0, alive: true }, // Concept B
        Node { label: 103, feature: 0.2, activation: 0.0, bias: 0.0, alive: true }, // Concept C
    ];

    let mut edge_arena: Vec<Edge> = vec![
        Edge { src: NodeId(0), dst: NodeId(1), relation: 5, weight: 0.5, alive: true, confidence: 1.0, next_in: 0, next_out: 0 },
        Edge { src: NodeId(0), dst: NodeId(2), relation: 6, weight: 0.5, alive: true, confidence: 1.0, next_in: 0, next_out: 0 },
    ];

    println!("✅ Mesh populated with 3 Conceptual Nodes and 2 Competing Edges.");
    println!("Commands: 'activate <0-2>' (stimulate a node), 'status' (view memory), 'exit'\n");

    loop {
        print!("mesh_user> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let command = input.trim();

        if command == "exit" {
            println!("Shutting down engine...");
            break;
        } else if command == "status" {
            println!("\n--- MESH CURRENT STATE ---");
            for (i, n) in node_arena.iter().enumerate() {
                println!("  Node [{}]: Label={}, Activation={:.2}", i, n.label, n.activation);
            }
            for (i, e) in edge_arena.iter().enumerate() {
                println!("  Edge [{}]: From Node {} to {}, Weight={:.2}", i, e.src.0, e.dst.0, e.weight);
            }
            println!("--------------------------\n");
        } else if command.starts_with("activate ") {
            let parts: Vec<&str> = command.split_whitespace().collect();
            if let Some(id_str) = parts.get(1) {
                if let Ok(idx) = id_str.parse::<usize>() {
                    if idx < node_arena.len() {
                        // Inject energy into the selected node
                        node_arena[idx].activation = 1.0;
                        println!("⚡ Injected 1.0 energy into Node [{}].", idx);
                        
                        // --- SIMULATING THE REWRITE LOOP logic Qwen built ---
                        // If we activate Node 0, it pours energy into its paths
                        if idx == 0 {
                            println!("🌊 Spreading activation traveling down outbound edges...");
                            node_arena[1].activation += edge_arena[0].weight * 0.5;
                            node_arena[2].activation += edge_arena[1].weight * 0.5;

                            println!("📉 Competitive Structural Inhibition rule triggered!");
                            println!("   (Edge 0 was used more, so Edge 1's weight decays to balance the node's budget)");
                            edge_arena[0].weight = (edge_arena[0].weight + 0.1).min(1.0);
                            edge_arena[1].weight = (edge_arena[1].weight - 0.1).max(0.0);
                        }
                    } else { println!("❌ Invalid Node ID."); }
                }
            }
        } else {
            println!("❓ Unknown command. Try 'activate 0', 'status', or 'exit'.");
        }
    }
}
