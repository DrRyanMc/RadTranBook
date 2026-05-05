import cProfile
import pstats
import sys

def main():
    import MG_IMC.test_marshak_wave_multigroup_powerlaw as mg_test
    import sys
    sys.argv = ["prog", "--groups", "2", "--Nmax", "5000", "--Ntarget", "2000", "--Nboundary", "2000", "--final-time", "0.05"]
    mg_test.main()

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        main()
    except Exception as e:
        print("Error:", e)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(30)
