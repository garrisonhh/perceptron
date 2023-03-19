const std = @import("std");

pub const Float = f32;

var rng = std.rand.DefaultPrng.init(0);

/// random float in the range [-1, 1)
fn randNeural() Float {
    return rng.random().float(Float) * 2.0 - 1.0;
}

fn sigmoid(x: Float) Float {
    return 1.0 / (1.0 + std.math.exp(-x));
}

fn sigmoid_derivative(x: Float) Float {
    const s: Float = sigmoid(x);
    return s * (1.0 - s);
}

fn activate(x: Float) bool {
    return x > 0.5;
}

fn encode(b: bool) Float {
    return if (b) 1 else 0;
}

/// a data entry encoding for a perceptron
fn PerceptronEntry(comptime INPUT_LEN: usize) type {
    return struct {
        const Self = @This();

        /// [0 .. INPUT_LEN] are input; [INPUT_LEN] is output
        const Set = std.StaticBitSet(INPUT_LEN + 1);

        bits: Set = Set.initEmpty(),

        pub fn getInput(self: *const Self, index: usize) bool {
            return self.bits.isSet(index);
        }

        pub fn getOutput(self: *const Self) bool {
            return self.bits.isSet(INPUT_LEN);
        }

        pub fn setInput(self: *Self, index: usize) void {
            self.bits.set(index);
        }

        pub fn setOutput(self: *Self) void {
            return self.bits.set(INPUT_LEN);
        }
    };
}

pub fn Perceptron(comptime INPUT_LEN: usize) type {
    return struct {
        const Self = @This();

        pub const Entry = PerceptronEntry(INPUT_LEN);

        weights: [INPUT_LEN]Float,
        bias: Float,

        pub fn initRandom() Self {
            var self = Self{
                .weights = undefined,
                .bias = randNeural(),
            };

            for (self.weights) |*n| n.* = randNeural();

            return self;
        }

        pub fn format(
            self: *const Self,
            comptime _: []const u8,
            _: std.fmt.FormatOptions,
            writer: anytype,
        ) @TypeOf(writer).Error!void {
            try writer.print(
                "perceptron({d:03.2} - {d:03.2})",
                .{self.weights, self.bias},
            );
        }

        pub fn forward(self: *const Self, inputs: []const Float) Float {
            std.debug.assert(inputs.len == INPUT_LEN);

            var weighted_sum = self.bias;
            for (inputs) |n, i| {
                weighted_sum += n * self.weights[i];
            }

            return sigmoid(weighted_sum);
        }

        pub fn backward(
            self: *Self,
            inputs: []const Float,
            expected: Float,
            actual: Float,
            learning_rate: Float,
        ) void {
            std.debug.assert(learning_rate > 0.0 and learning_rate < 1.0);

            const err = expected - actual;

            // represents sigmoid derivative on model output, multiplied by the
            // error (pretty sure this is MSE)
            const gradient = err * actual * (1 - actual);
            const adjustment = gradient * learning_rate;

            for (self.weights) |*weight, i| {
                weight.* += adjustment * inputs[i];
            }

            self.bias += adjustment;
        }

        pub const TrainingResults = struct {
            accuracy: f64,
        };

        /// an iteration of training over a dataset
        pub fn train(
            self: *Self,
            entries: []const Entry,
            learning_rate: Float,
        ) TrainingResults {
            var num_correct: usize = 0;

            // forward
            var buf: [INPUT_LEN]Float = undefined;
            const inputs = buf[0..];
            for (entries) |*entry| {
                // set up entry
                var i: usize = 0;
                while (i < INPUT_LEN) : (i += 1) {
                    inputs[i] = encode(entry.getInput(i));
                }

                const expect_bool = entry.getOutput();
                const expect = encode(expect_bool);

                // run perceptron training cycle
                const output = self.forward(inputs);
                self.backward(inputs, expect, output, learning_rate);

                // stats
                if (expect_bool == activate(output)) {
                    num_correct += 1;
                }
            }

            // collect stats
            // TODO technically this is inaccurate, since I'm modifying the
            // model on each iteration. I should do all of the forwarding at
            // once and then all of the back prop at once to get an accurate
            // accuracy measurement
            const accuracy = @intToFloat(f64, num_correct) /
                             @intToFloat(f64, entries.len);

            return TrainingResults{
                .accuracy = accuracy,
            };
        }
    };
}
