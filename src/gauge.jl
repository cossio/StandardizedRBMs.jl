function rescale_hidden_activations!(rbm::StandardizedRBM)
    if rescale_activations!(rbm.hidden, rbm.scale_h)
        rbm.offset_h ./= rbm.scale_h
        rbm.scale_h ./= rbm.scale_h
        return true
    end
    return false
end
