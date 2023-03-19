const std = @import("std");
const stdout = std.io.getStdOut().writer();
const perceptron = @import("perceptron.zig");
const Perceptron = perceptron.Perceptron;

const LogicPerceptron = Perceptron(2);
const LogicEntry = LogicPerceptron.Entry;

pub fn main() !void {
    // create dataset
    var dataset: [4]LogicEntry = undefined;
    for (dataset) |*entry, i| {
        const a = (i >> 1) & 1 > 0;
        const b = (i >> 0) & 1 > 0;
        const out = a and b;

        entry.* = LogicEntry{};
        if (a) entry.setInput(0);
        if (b) entry.setInput(1);
        if (out) entry.setOutput();
    }

    // train
    const TRAIN_ITERATIONS = 1000;
    const LEARNING_RATE = 0.5;

    var p = LogicPerceptron.initRandom();

    try stdout.writeAll("[training]\n");

    var i: usize = 0;
    while (i < TRAIN_ITERATIONS) : (i += 1) {
        const res = p.train(&dataset, LEARNING_RATE);
        const acc_percent = res.accuracy * 100;

        try stdout.print(
            "{d:6}) {d: >4.2}% accuracy - {}\n",
            .{i, acc_percent, p},
        );
    }
}
