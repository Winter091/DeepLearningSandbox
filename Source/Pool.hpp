#ifndef __POOL_H__
#define __POOL_H__

#include <vector>
#include <cstdint>


struct PoolElement
{
    uint8_t Target;
    std::vector<float> Features;
};  


class Pool
{
public:
    Pool(const char* imagesFile, const char* labelsFile);

    template <typename T>
    Pool(T&& elements)
        : m_elements(std::forward<T>(elements))
        , m_size(m_elements.size())
    {}

    std::size_t GetSize() const { return m_size; }
    const std::vector<PoolElement>& GetElements() const { return m_elements; }

    Pool TakeRandom(std::size_t amount) const;

private:
    std::vector<PoolElement> m_elements;
    std::size_t m_size = 0;

private:
    void ParseLabelsFile(const char* labelsFile);
    void ParseImagesFile(const char* imagesFile);
    uint32_t ReverseByteOrder(uint32_t num);
};

#endif