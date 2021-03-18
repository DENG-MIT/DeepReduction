
function updateyaml(mech, p)
    yaml = YAML.load_file(mech)
    n_species = length(yaml["phases"][1]["species"])
    n_reactions = length(yaml["reactions"])
    species_names = yaml["phases"][1]["species"]
    elements = yaml["phases"][1]["elements"]
    n_elements = length(elements)

    _p = reshape(p, n_reactions, 3)

    for i = 1:n_reactions
        p_vec = _p[i, :]
        fA = exp(p_vec[1])
        fb = p_vec[2]
        fEa = p_vec[3]
        reaction = yaml["reactions"][i]
        if haskey(reaction, "type")
            if (reaction["type"] == "falloff")
                yaml["reactions"][i]["low-P-rate-constant"]["A"] *= fA
                yaml["reactions"][i]["low-P-rate-constant"]["b"] += fb
                yaml["reactions"][i]["low-P-rate-constant"]["Ea"] += fEa
                yaml["reactions"][i]["high-P-rate-constant"]["A"] *= fA
                yaml["reactions"][i]["high-P-rate-constant"]["b"] += fb
                yaml["reactions"][i]["high-P-rate-constant"]["Ea"] += fEa
            else
                yaml["reactions"][i]["rate-constant"]["A"] *= fA
                yaml["reactions"][i]["rate-constant"]["b"] += fb
                yaml["reactions"][i]["rate-constant"]["Ea"] += fEa
            end
        else
            yaml["reactions"][i]["rate-constant"]["A"] *= fA
            yaml["reactions"][i]["rate-constant"]["b"] += fb
            yaml["reactions"][i]["rate-constant"]["Ea"] += fEa
        end
    end

    YAML.write_file(mech[1:end-5] * "_op.yaml", yaml)
end
