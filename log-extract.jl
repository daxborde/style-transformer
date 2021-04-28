print("d_adv_loss, f_slf_loss, f_cyc_loss, f_adv_loss\n")
open("loss_log.txt", "r") do f
    for line in readlines(f)
        m = match(r"(?:\S+ ){3}([\d.]+)(?:\S* ){3}([\d.]+)(?:\S* ){3}([\d.]+)(?:\S* ){3}([\d.]+)", line)
        mnums = map(x->parse(Float64,x), m.captures)

        print("$(mnums[1]), $(mnums[2]), $(mnums[3]), $(mnums[4])\n")
    end
end