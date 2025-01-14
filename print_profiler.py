import pstats

prof = pstats.Stats('profiler_output.prof')

prof.strip_dirs()  
prof.sort_stats('cumulative') 
prof.print_stats(20) 