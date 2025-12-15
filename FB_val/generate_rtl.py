from amaranth import *
from amaranth.back import verilog

class HDCAccelerator(Elaboratable):
    def __init__(self, dimension=10000):
        # PATENT CLAIM: D >= 10,000
        self.dimension = dimension
        
        # HARDWARE INTERFACE
        # 10,000 parallel input pins for Query and Memory
        self.query_in = Signal(dimension)
        self.memory_in = Signal(dimension)
        
        # Output: Hamming Distance
        # 14 bits is required to hold a value up to 10,000 (2^14 = 16,384)
        self.hamming_dist = Signal(14)

    def elaborate(self, platform):
        m = Module()

        # STEP 1: PARALLEL XOR (10,000 Gates in one line)
        xor_result = Signal(self.dimension)
        m.d.comb += xor_result.eq(self.query_in ^ self.memory_in)

        # STEP 2: BALANCED ADDER TREE (The "Popcount" Engine)
        # Reduces 10,000 bits to 1 number in just ~14 logic levels.
        
        # Convert the XOR vector into a list of individual 1-bit signals
        signals = [xor_result[i] for i in range(self.dimension)]
        
        # Iteratively sum pairs until one signal remains
        while len(signals) > 1:
            next_layer = []
            for i in range(0, len(signals), 2):
                if i + 1 < len(signals):
                    next_layer.append(signals[i] + signals[i+1])
                else:
                    next_layer.append(signals[i]) # Carry odd bit
            signals = next_layer

        # The last remaining signal is the total Hamming Distance
        m.d.comb += self.hamming_dist.eq(signals[0])

        return m

if __name__ == "__main__":
    print("Generating PRODUCTION RTL for HDC Accelerator (D=10,000)...")
    
    top = HDCAccelerator(dimension=10000)
    
    # Generate the full hardware description
    with open("hdc_core_10k.v", "w") as f:
        f.write(verilog.convert(top, name="HDC_Accelerator_10k", ports=[
            top.query_in, 
            top.memory_in, 
            top.hamming_dist
        ]))
        
    print("[SUCCESS] Hardware Description generated: 'hdc_core_10k.v'")
    print("Specs: 10,000-bit XOR Array + 14-Stage Adder Tree")